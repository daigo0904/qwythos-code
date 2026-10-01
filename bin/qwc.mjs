#!/usr/bin/env node
// Qwythos Code — 入り口。起動オプションの解釈と、対話ループ。
import readline from 'node:readline';
import path from 'node:path';
import process from 'node:process';
import fs from 'node:fs';
import { loadConfig, saveConfig, HOME_DIR } from '../src/config.mjs';
import { checkServer, listModels, adaptToModel, pickBestModel, checkGpuFit } from '../src/ollama.mjs';
import { PermissionManager, saysYes } from '../src/permissions.mjs';
import { Agent, estimateTokens } from '../src/agent.mjs';
import { contextLine } from '../src/ctxcost.mjs';
import { TOOLS, activeTools, setMcpTools, KEY_HELP } from '../src/tools.mjs';
import { startMcp, stopMcp, loadMcpConfig } from '../src/mcp.mjs';
import { loadApiKey } from '../src/web.mjs';
import { resolveMentions, buildMentionBlock } from '../src/mentions.mjs';
import { anyServerAvailable, serverStatus, stopAll as stopLsp } from '../src/lsp.mjs';
import { loadCommands, renderCommand, isReserved, BUILTIN_COMMANDS } from '../src/commands.mjs';
import { makeCompleter } from '../src/complete.mjs';
import { createPasteBuffer, attachBracketedPaste } from '../src/paste.mjs';
import { undoLastTurn, sessionChanges, canUndo, MAX_ENTRIES } from '../src/edits.mjs';
import {
  isAvailable as browserAvailable,
  login as browserLogin,
  savedSites,
  forgetAll,
  INSTALL_HELP as BROWSER_HELP
} from '../src/browser.mjs';
import { newSessionId, saveSession, listSessions, loadSession, latestSessionForRoot } from '../src/session.mjs';
import { loadHarness, undoHarness, describeHarness } from '../src/harness.mjs';
import {
  c, line, out, banner, info, warn, error, success, renderTodos,
  sendDisplayToStderr, Spinner, renderDiff, formatTiming
} from '../src/ui.mjs';
import { serve, requestTool, emit, ready, turnEnd } from '../src/embed.mjs';
import { createEventLog } from '../src/events.mjs';
import { EFFORT_LEVELS, EFFORT_ORDER, DEFAULT_EFFORT, normalizeEffort } from '../src/effort.mjs';

const VERSION = '0.2.0';

// ── 起動オプション ──────────────────────────────────────────
function parseArgs(argv) {
  const opts = { prompt: null, resume: null, cwd: null, overrides: {}, listSessions: false };
  const rest = [];

  /**
   * 数で受け取る旗を、受け取ったその場で確かめる。
   *
   * ■ 素通ししていたせいで何が起きるか
   *   `--ctx 32k` や値の書き忘れは `Number()` が NaN を返し、そのまま設定に入っていた。
   *   すると文脈のしきい値（numCtx × compactAtRatio・agent.mjs）も NaN になる。
   *   **NaN との大小はいつも false** なので「まだ短い」と一度も判定されず、
   *   会話が9件を超えた時点から**毎ターン要約が走る**。
   *   毎ターンの圧縮は履歴を書き換えるので、実測で 310秒 → 520秒（1.68倍）に落ちる経路そのもの。
   *
   * ■ しかも黙って居座る
   *   その状態で `/save` すると、JSON に NaN は書けないので `null` が残る。
   *   `null` は既定値を上書きするので、**次に起動してもずっと毎ターン圧縮のまま**になる。
   *   画面には最後まで何も出ない。だから通す前に止める。
   */
  const asNumber = (flag, raw, { integer = true, min = 1, max = Infinity } = {}) => {
    const v = Number(raw);
    const ok = Number.isFinite(v) && v >= min && v <= max && (!integer || Number.isInteger(v));
    if (!ok) {
      const range = max === Infinity ? `${min} 以上` : `${min}〜${max}`;
      error(`${flag} には${integer ? '整数' : '数'}を指定してください（${range}）。受け取った値: ${raw === undefined ? '（なし）' : raw}`);
      process.exit(1);
    }
    return v;
  };

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    switch (a) {
      case '-h': case '--help': opts.help = true; break;
      case '-v': case '--version': opts.version = true; break;
      case '-p': case '--print': opts.prompt = next(); break;
      case '--events': opts.events = next(); break;
      case '-m': case '--model': opts.overrides.model = next(); break;
      case '--host': opts.overrides.host = next(); break;
      case '--ctx': opts.overrides.numCtx = asNumber('--ctx', next()); break;
      case '--temp': opts.overrides.temperature = asNumber('--temp', next(), { integer: false, min: 0, max: 2 }); break;
      case '--steps': opts.overrides.maxSteps = asNumber('--steps', next()); break;
      case '--yolo': case '--dangerously-skip-permissions': opts.overrides.autoApprove = true; break;
      // 「確認なし」を既定として保存してあるとき、その回だけ確認ありに戻すため。
      // 保存できるようにした以上、抜け道が無いと危ない作業のときに困る。
      case '--confirm': case '--no-yolo': opts.overrides.autoApprove = false; break;
      case '--accept-edits': opts.overrides.acceptEdits = true; break;
      case '--no-think': opts.overrides.effort = 'off'; break;
      case '--effort': opts.overrides.effort = next(); break;
      case '--show-thinking': opts.overrides.showThinking = 'full'; break;
      case '--quiet-thinking': opts.overrides.showThinking = 'off'; break;
      case '--allow-outside': opts.overrides.allowOutsideRoot = true; break;
      case '--no-net': opts.overrides.net = false; break;
      case '--plan': opts.overrides.planMode = true; break;
      case '--chat': opts.overrides.chatMode = true; break;
      case '--embed': opts.embed = true; break;
      case '--cwd': opts.cwd = next(); break;
      case '--resume': opts.resume = argv[i + 1] && !argv[i + 1].startsWith('-') ? next() : 'last'; break;
      case '--sessions': opts.listSessions = true; break;
      default:
        if (a.startsWith('-')) {
          error(`知らないオプションです: ${a}`);
          process.exit(1);
        }
        rest.push(a);
    }
  }
  if (!opts.prompt && rest.length) opts.prompt = rest.join(' ');
  return opts;
}

function showHelp() {
  line(`
${c.bold('Qwythos Code')} v${VERSION} — ローカルのモデルで動く自律コーディングエージェント

${c.bold('使い方')}
  qwc                          対話モードで起動する
  qwc -p "テストを直して"       一回だけ実行して終了する
  qwc --resume                 直前の会話の続きから始める

${c.bold('オプション')}
  -p, --print <文>       一回だけ実行して結果を出す（自動化向け）
      --events <file>    -p の走りを codex exec --json と同じ形の JSONL で残す
  -m, --model <名前>     使うモデル（既定: gemma4:26b）
      --host <URL>       Ollama の場所（既定: http://localhost:11434）
      --ctx <数>         文脈の広さ（既定: 32768）
      --temp <数>        創造性の高さ 0〜1（既定: 0.3）
      --steps <数>       1回のお願いで許すツール実行の上限（既定: 200）
      --yolo             確認を全部飛ばす（--dangerously-skip-permissions も同じ）
      --confirm          その回だけ確認ありに戻す（--no-yolo も同じ）
      --accept-edits     書き換えだけ確認なしにする（コマンドとネットは確認する）
      --no-think         思考モードを切る（速くなるが精度は落ちる）
      --effort <段階>    考える深さ（off / low / medium / high・既定 medium）
      --show-thinking    考えている内容を全部表示する
      --quiet-thinking   考えている様子を表示しない
      --cwd <フォルダ>   作業フォルダを指定する
      --allow-outside    作業フォルダの外も触れるようにする
      --no-net           ネット接続を切る（検索とページ取得の道具を渡さない）
      --plan             計画モードで始める（まず調べて方針を出す。書き換えない）
      --chat             雑談モードで始める（普通に話す。書き換えない）
      --resume [ID]      前の会話を読み込む（IDなしなら直近）
      --sessions         保存済みの会話を一覧する
  -h, --help             このヘルプ
  -v, --version          バージョン

${c.bold('対話中に使えるコマンド')}
  /help /clear /compact /model /think /effort /yolo /accept /plan /chat /tools /stats /files /diff /undo /refine /init /exit
  /login /logins /logout   ログインが要るサイトを読めるようにする
`);
}

function showSlashHelp() {
  line(`
${c.bold('  対話中のコマンド')}
  ${c.cyan('/help')}      この一覧
  ${c.cyan('/clear')}     会話をまっさらにする（作業フォルダはそのまま）
  ${c.cyan('/compact')}   会話をいますぐ要約して短くする
  ${c.cyan('/model')}     モデルを見る・切り替える（例: /model qwen3:14b-q4_K_M）
  ${c.cyan('/think')}     考えさせるか＋考えている様子の出し方（on / off / compact / full / quiet）
  ${c.cyan('/effort')}    考える深さ（off / low / medium / high）
  ${c.cyan('/yolo')}      確認あり／なしを切り替える（全部飛ばす）
  ${c.cyan('/accept')}    書き換えだけ確認なしにする（コマンドとネットは確認する）
  ${c.cyan('/plan')}      計画モードの出入り（まず調べて方針を出す。書き換えない）
  ${c.cyan('/chat')}      雑談モードの出入り（ずっと雑談。ふだんは自動で見分けるので不要）
  ${c.cyan('/todo')}      いまのやることリストを見る
  ${c.cyan('/commands')}  自分で作ったコマンドの一覧（.qwythos/commands/*.md）
  ${c.cyan('/tools')}     使える道具の一覧
  ${c.cyan('/login')}     サイトに手でログインして、その状態を保存する（例: /login github.com）
  ${c.cyan('/logins')}    ログイン状態が残っているサイトの一覧
  ${c.cyan('/logout')}    保存したログイン状態をすべて消す
  ${c.cyan('/stats')}     使ったトークン数などの記録
  ${c.cyan('/files')}     このセッションで書き換えたファイル
  ${c.cyan('/diff')}      このセッションの変更を差分でまとめて見る（例: /diff src/app.js）
  ${c.cyan('/undo')}      直前のお願いでした書き換えを、まとめて元に戻す
  ${c.cyan('/refine')}    いまのやり取りから、覚えておくことを直す（/refine list ・ /refine undo）
  ${c.cyan('/init')}      プロジェクトを調べて QWYTHOS.md を作る
  ${c.cyan('/save')}      いまの設定を次回以降の既定にする
  ${c.cyan('/exit')}      終了（Ctrl+D でも同じ）

  ${c.gray('Tab で補完できます（/コマンド名 ・ @ファイル名 ・ /model のモデル名）。')}
  ${c.gray('複数行を貼り付けると、1つの依頼としてまとめて受け取ります。')}
`);
}

/**
 * 思考モードが、いまどうなっているかを見せる。
 *
 * ■ なぜ要るか
 *   ここには**別々の2つ**がある。「考えさせるかどうか」と「考えている様子を画面に出すかどうか」。
 *   どちらも見えないので、切り替えたつもりで切り替わっていないのか、
 *   切り替わったが画面に出ていないだけなのかが分からなかった。
 *
 *   さらに、希望（thinkPreference）どおりになるとは限らない。
 *   モデルが思考モードを持たなければ、実際に送る値（think）は false に落ちる。
 *   「入れました」とだけ出すと、その場で嘘になる。**実際に効いている値**を出す。
 */
async function showThinkState(config) {
  const label = {
    off: '出さない',
    compact: '1行だけ出す',
    full: '全部出す'
  }[config.showThinking] ?? config.showThinking;

  // 見出しの幅をそろえる。全角なので、文字数ではなく見た目の幅で詰める
  line();
  line(`  ${c.gray('考えさせる　　')} ${config.think ? c.green('はい') : c.gray('いいえ')}`);
  line(
    `  ${c.gray('考えている様子')} ` +
      (config.think
        ? config.showThinking === 'off'
          ? c.gray(label)
          : c.green(label)
        : c.gray(`${label}（考えさせていないので、いまは関係ありません）`))
  );

  // 希望と実際がずれているときだけ、理由を添える
  if (config.thinkPreference && !config.think) {
    line(`  ${c.yellow('※')} ${config.model} は思考モードを持たないので、考えさせる指定は効きません。`);
  }
  if (config.think && config.showThinking === 'off') {
    line(`  ${c.gray('※ 考えてはいますが、様子は画面に出していません（/think compact で出せます）。')}`);
  }
  line(`  ${c.gray('変えるには: /think on | off | compact | full | quiet')}`);
  line();
}

/**
 * いまの「考える深さ」を出す。
 *
 * **効いていないものを「効いています」と書かないこと。** `/think` で一度これをやっている
 * （compact は表示を変えるだけなのに、思考を浅くするものだと読める名前だった）。
 *
 * 深さの効かせ方は2つあり、モデルによって効く側が違う。
 *   1. ollama の think 段階  … 段階を持つモデルだけ（gemma4:26b は無視する。2026-09-03 実測）
 *   2. system プロンプトの一文 … 言うことを聞くモデルだけ
 * どちらが今このモデルに効いているかは断言できないので、**送っている中身をそのまま見せる**。
 */
function showEffortState(config) {
  const cur = normalizeEffort(config.effort) ?? DEFAULT_EFFORT;
  // 全角は2桁ぶんの幅を取る。padEnd は文字数で数えるので、そのままだと列がずれる
  //（`/think` の表示でも同じ手当てをしている）。
  const wide = (t) => [...t].reduce((w, ch) => w + (/[^\x00-\xff]/.test(ch) ? 2 : 1), 0);
  const pad = (t, w) => t + ' '.repeat(Math.max(0, w - wide(t)));

  line();
  for (const key of EFFORT_ORDER) {
    const lv = EFFORT_LEVELS[key];
    const here = key === cur;
    const mark = here ? c.green('●') : c.gray('○');
    const name = pad(key, 8);
    const label = pad(lv.label, 10);
    line(`  ${mark} ${here ? c.green(name) : c.gray(name)}${c.gray(label)}${c.gray(lv.hint)}`);
  }
  line();
  line(`  ${c.gray('ollama に送る think')} ${c.cyan(String(config.think))}`);
  const depth = EFFORT_LEVELS[cur].directive;
  line(`  ${c.gray('指示文に足す一文　')} ${depth ? c.cyan(depth) : c.gray('（なし）')}`);
  if (config.thinkPreference && config.think === false) {
    line(`  ${c.yellow('※')} ${config.model} は思考モードを持たないので、深さの指定は効きません。`);
  }
  line(`  ${c.gray('変えるには: /effort off | low | medium | high')}`);
  line();
}

// ── 打った依頼の履歴（↑ で呼び出せるようにする） ──────────────
//
// 終了しても残す。同じ作業を翌日に続けることが多く、毎回打ち直すのは無駄なため。
// 残すのは**本人が打った依頼だけ**で、承認の y/n/a は残さない。
const HISTORY_MAX = 200;
const HISTORY_FILE = path.join(HOME_DIR, 'history');

function loadHistory() {
  try {
    return fs
      .readFileSync(HISTORY_FILE, 'utf8')
      .split('\n')
      .map((l) => l.trimEnd())
      .filter((l) => l !== '')
      .slice(-HISTORY_MAX);
  } catch {
    return [];
  }
}

/** 打った依頼を控える。すぐ書くので、落ちても直前まで残る */
function rememberInput(text) {
  const value = String(text ?? '').trim();
  if (!value || value.includes('\n')) return;
  try {
    const kept = loadHistory().filter((l) => l !== value);
    kept.push(value);
    fs.mkdirSync(HOME_DIR, { recursive: true });
    fs.writeFileSync(HISTORY_FILE, `${kept.slice(-HISTORY_MAX).join('\n')}\n`, 'utf8');
  } catch {
    // 履歴が書けなくても作業は続けられる。ここで止めない
  }
}

/** 承認の答えなど、残したくない行を履歴から抜く */
function dropFromHistory(text) {
  const value = String(text ?? '');
  if (!rl || !Array.isArray(rl.history)) return;
  // readline は新しいものを先頭に積む。いま入ったものだけを見る
  if (rl.history[0] === value) rl.history.shift();
}

// ── 対話の入力受け付け ──────────────────────────────────────
let rl = null;
let bracketedPaste = null;
let pendingAsk = null;
let inputClosed = false;
const inputQueue = [];

function createReadline({ root, models = () => [] } = {}) {
  rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: Boolean(process.stdin.isTTY),
    historySize: HISTORY_MAX,
    completer: root
      ? makeCompleter({
          root,
          // 対話中に足したコマンドも拾えるよう、そのつど読み直す
          commandNames: () => [...BUILTIN_COMMANDS, ...loadCommands(root).keys()],
          modelNames: models
        })
      : undefined
  });

  // 前に打った依頼を、↑ で呼び出せるようにする。
  // readline の history は「新しいものが先頭」なので、逆順に入れる。
  if (process.stdin.isTTY && Array.isArray(rl.history)) {
    rl.history.push(...loadHistory().reverse());
  }

  // 作業中に打たれた行も取りこぼさないよう、いったん溜めておく
  const deliverLine = (text) => {
    if (pendingAsk) {
      const resolve = pendingAsk;
      pendingAsk = null;
      resolve(text);
    } else {
      inputQueue.push(text);
    }
  };

  // 貼り付けの最中の改行を、行の確定として扱わせない（paste.mjs）。
  // これが無いと、複数行を貼っただけでエンターを押さずに依頼が飛ぶ。
  bracketedPaste = process.stdin.isTTY
    ? attachBracketedPaste(rl, {
        output: process.stdout,
        onNote: (msg) => {
          line();
          warn(msg);
          // 書いたぶん画面がずれるので、入力欄を引き直す（打ちかけの文は残す）
          rl.prompt(true);
        }
      })
    : null;
  if (bracketedPaste) {
    bracketedPaste.enable();
    // 印を出させたまま終わると、そのあとのシェルに残る。必ず消す
    process.on('exit', () => bracketedPaste?.disable());
  }

  // 印に対応していない端末むけの保険。
  // 続けざまに届いた行は、貼り付けとみなして1つにまとめる（paste.mjs）
  const paste = createPasteBuffer(deliverLine, { enabled: Boolean(process.stdin.isTTY) });
  // 入力欄の札（[貼り付け1: 12行]）を、貼られた中身に戻してから受け取る
  rl.on('line', (text) => paste.push(bracketedPaste ? bracketedPaste.expand(text) : text));

  // Ctrl+D や入力の終わりで、待っている質問を解いてループを抜けられるようにする
  rl.on('close', () => {
    bracketedPaste?.disable();
    // 溜めたままの行を捨てない。貼り付けた直後に Ctrl+D を押されると、
    // 15ミリ秒の待ちに入っていたぶんが、そのまま消える
    paste.flush();
    inputClosed = true;
    if (pendingAsk) {
      const resolve = pendingAsk;
      pendingAsk = null;
      resolve(null);
    }
  });
  return rl;
}

function showPrompt(question) {
  if (rl && process.stdin.isTTY) {
    // 道具で走らせた子プロセス（vim など）が終わると、貼り付けの印が消えることがある。
    // 入力を待つたびにかけ直す。7バイト書くだけなので、毎回でも損はない
    bracketedPaste?.enable();
    rl.setPrompt(question);
    rl.prompt();
  } else {
    out(question);
  }
}

/**
 * 1行受け取る。
 *
 * `remember` を立てるのは、本人が打った**依頼**だけ。
 * 承認の y/n/a まで残すと、↑ を押しても "y" しか出てこなくなり、履歴が使い物にならない
 * （実際そうなっていた）。残さないものは、読んだ直後に履歴から抜く。
 */
function ask(question, { remember = false } = {}) {
  return new Promise((resolve) => {
    // 対話できない状況（-p など）では null を返す＝確認は「やめる」扱い
    if (!rl) return resolve(null);

    const deliver = (value) => {
      // パイプ入力は画面に出ないので、記録として自分で書き出す
      if (value !== null && !process.stdin.isTTY) line(value);
      // 貼り付けをまとめたときは、そう言っておく。
      // 黙ってまとめると、1行しか送られていないのか全部届いたのか分からない。
      if (remember && typeof value === 'string' && value.includes('\n')) {
        info(`${value.split('\n').length} 行を、1つの依頼としてまとめて受け取りました。`);
      }
      if (!remember) dropFromHistory(value);
      else rememberInput(value);
      resolve(value);
    };

    showPrompt(question);
    if (inputQueue.length) return deliver(inputQueue.shift());
    if (inputClosed) return resolve(null);
    pendingAsk = deliver;
  });
}

// ── 別のアプリの中で動くときの本体 ──────────────────────────
//
// 相手から届く hello で、こちらの道具立てが決まる。
// 相手が持っていない道具はモデルに見せない（呼べない道具を見せない、はここでも同じ）。
async function runEmbedded({ config, root }) {
  let agent = null;
  let workRoot = root;

  await serve({
    onHello(message) {
      // 作業フォルダは相手が決める。あちらの利用者が選んだ場所がすべて。
      if (typeof message.root === 'string' && message.root) {
        workRoot = path.resolve(message.root);
        try {
          process.chdir(workRoot);
        } catch {
          emit.error(`指定されたフォルダに移動できません: ${workRoot}`);
        }
      }
      if (typeof message.model === 'string' && message.model) config.model = message.model;

      // 道具の実体は相手にある。呼び出しはそちらへ回す。
      config.hostToolNames = Array.isArray(message.tools) ? message.tools : [];
      config.hostTools = requestTool;
      // 確認は持ち主の仕事。こちらで聞くと、同じことを二度聞かれることになる。
      config.autoApprove = true;
      config.onContentDelta = (text) => emit.text(text);
      config.onTodos = (todos) => emit.todos(todos);

      agent = new Agent({ config, root: workRoot, permissions: null, onSave: () => {} });
      ready(activeTools(config).map((t) => t.name));
    },

    async onUser(message) {
      if (!agent) {
        emit.error('先に hello を送ってください。');
        turnEnd('');
        return;
      }
      const result = await agent.runTurn(String(message.text ?? ''));
      // 最後にモデルが言ったことを、そのまま返す
      let answer = '';
      for (let i = agent.messages.length - 1; i >= 0; i--) {
        const m = agent.messages[i];
        if (m.role === 'assistant' && typeof m.content === 'string' && m.content.trim()) {
          answer = m.content.trim();
          break;
        }
      }
      turnEnd(answer, result?.interrupted);
    },

    onInterrupt() {
      agent?.interrupt();
    },

    onGone() {
      agent?.interrupt();
    }
  });

  // 相手がいなくなったら、こちらも終わる。
  // 待っている通信が残っていると輪が抜けきらないことがあるので、ここで断ち切る。
  process.exit(0);
}

// ── 本体 ────────────────────────────────────────────────────
async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help) return showHelp();
  if (opts.version) return line(`qwc ${VERSION}`);

  if (opts.listSessions) {
    const sessions = listSessions(20);
    if (!sessions.length) return info('保存された会話はまだありません。');
    line();
    for (const s of sessions) {
      line(`  ${c.cyan(s.id)}  ${c.gray(`${s.root || '?'} — ${s.firstUser || ''}`)}`);
    }
    line();
    return;
  }

  // 別のアプリの中でエンジンとして動くときは、標準出力を JSON のやりとりに使う。
  // 色つきの表示が混ざると相手が読めなくなるので、いちばん先に行き先を変える
  // （ここから下では、つなぎ先の確認やモデルの選び直しが info/error を出しうる）。
  if (opts.embed) sendDisplayToStderr();

  const config = { ...loadConfig(), ...opts.overrides };
  const root = path.resolve(opts.cwd || process.cwd());
  try {
    process.chdir(root);
  } catch {
    error(`フォルダに移動できません: ${root}`);
    process.exit(1);
  }

  // Ollama が動いているか確認する
  const health = await checkServer(config);
  if (!health.ok) {
    error(`Ollama につながりません (${config.host})。`);
    info('別のターミナルで `ollama serve` を実行してから、もう一度お試しください。');
    process.exit(1);
  }
  // 一覧はこの下の「載りきるか」の判定でも使うので、try の外に出しておく
  let models = [];
  try {
    models = await listModels(config);
    // Ollama では「名前だけ」は「名前:latest」と同じ扱いになる
    const wanted = config.model.includes(':') ? config.model : `${config.model}:latest`;
    let found = models.includes(config.model) || models.includes(wanted);

    // 量子化の接尾辞まで書いていない場合（qwen3:14b → qwen3:14b-q4_K_M）は前方一致で拾う
    if (!found) {
      const near = models.filter((m) => m.startsWith(`${config.model}-`) || m.startsWith(`${config.model}:`));
      if (near.length === 1) {
        info(`${config.model} は見つからないので ${near[0]} を使います。`);
        config.model = near[0];
        found = true;
      } else if (near.length > 1) {
        error(`${config.model} に当てはまるモデルが複数あります: ${near.join(', ')}`);
        info('どれを使うか -m で指定してください。');
        process.exit(1);
      }
    }

    if (!found) {
      // 無いからといって、いきなり終わらせない。
      // 入っているものから妥当なものを選んで、そのまま使えるようにする。
      // -m で明示されたときだけは、黙ってすり替えず本人に決めてもらう。
      const substitute = opts.overrides.model ? null : pickBestModel(models);
      if (substitute) {
        info(`"${config.model}" は入っていないので ${substitute} を使います（-m で変えられます）。`);
        config.model = substitute;
        found = true;
      } else {
        error(`モデル "${config.model}" が見つかりません。`);
        info(`使えるモデル: ${models.join(', ') || '（1つもありません）'}`);
        info(`取得するなら: ollama pull ${config.model}`);
        process.exit(1);
      }
    }
  } catch (err) {
    warn(`モデル一覧を確認できませんでした: ${err.message}`);
  }

  // GPU に載りきるかを先に確かめる。
  //
  // 載りきらないと、はみ出した分が CPU 側で動いて極端に遅くなるが、画面には何も出ない。
  // 「今日はなぜか遅い」で終わらせないために、ここで実際に測って、駄目なら軽いほうに落とす。
  // -m で明示されたときは落とさない。本人が選んだものを黙ってすり替えないため。
  if (config.autoDowngrade && !opts.overrides.model && config.lightModel && config.lightModel !== config.model) {
    const spinner = new Spinner(`${config.model} を読み込んでいます`).start();
    const fit = await checkGpuFit(config);
    spinner.stop();
    if (!fit.ok && models.includes(config.lightModel)) {
      const pct = Math.round(fit.onGpu * 100);
      warn(`${config.model} は GPU に ${pct}% しか載りませんでした（残りは CPU 側で動くため極端に遅くなります）。`);
      info(`${config.lightModel} に切り替えます。GPU の空きが戻れば、次回はそのまま ${config.model} を使います。`);
      info(`このまま使いたいときは -m ${config.model} を付けてください。`);
      config.model = config.lightModel;
    }
  }

  // モデルの能力に設定を合わせる（思考モードの有無など）
  const adapted = await adaptToModel(config);
  for (const note of adapted.notes || []) {
    if (note.level === 'error') error(note.text);
    else info(note.text);
  }
  if (adapted.canTools === false) {
    info('ツール呼び出しに対応したモデルを指定してください（例: -m gemma4:26b）。');
    process.exit(1);
  }

  // 外の道具（MCP）をつなぐ。
  // 設定が無ければ何も起きないので、立ち上げが遅くなるのは使う人だけ。
  {
    const { tools: mcp, notes } = await startMcp(root);
    if (mcp.length) {
      setMcpTools(mcp);
      info(`外の道具を ${mcp.length} 個つなぎました（${mcp.map((t) => t.name).join(', ')}）。`);
    }
    for (const note of notes) warn(note);
  }

  // ── 別のアプリの中でエンジンとして動く ──────────────────────
  //
  // 道具の実体は相手が持っている。こちらは輪だけを回して、
  // 「これを実行してほしい」と頼み、結果を受け取って次の一手を考える。
  // 相手の作法（権限・記録・確認）に手を出さないので、あちらの約束は壊れない。
  if (opts.embed) {
    await runEmbedded({ config, root });
    return;
  }

  const isInteractive = !opts.prompt && !opts.embed;
  if (isInteractive) createReadline({ root, models: () => models });

  const permissions = new PermissionManager(config, ask);
  const sessionId = newSessionId();

  const agent = new Agent({
    config,
    root,
    permissions,
    onSave: () => {
      saveSession(sessionId, {
        id: sessionId,
        root,
        model: config.model,
        updatedAt: new Date().toISOString(),
        messages: agent.messages.filter((m) => m.role !== 'system')
      });
    }
  });

  // 前の会話を読み込む
  if (opts.resume) {
    const id = opts.resume === 'last' ? latestSessionForRoot(root) : opts.resume;
    const data = id ? loadSession(id) : null;
    if (data) {
      agent.restore(data.messages);
      info(`前の会話を読み込みました (${id}, ${data.messages.length} 件)。`);
    } else {
      warn('読み込める会話が見つかりませんでした。新しく始めます。');
    }
  }

  // Ctrl+C の扱い
  let lastSigint = 0;
  const onSigint = () => {
    if (agent.running) {
      agent.interrupt();
      if (pendingAsk) {
        const resolve = pendingAsk;
        pendingAsk = null;
        out('\n');
        resolve('n');
      }
      return;
    }
    const now = Date.now();
    if (now - lastSigint < 1500) {
      line();
      info('またどうぞ。');
      process.exit(0);
    }
    lastSigint = now;
    line();
    info('終了するにはもう一度 Ctrl+C（または /exit）。');
    if (rl) rl.prompt();
  };
  process.on('SIGINT', onSigint);
  if (rl) rl.on('SIGINT', onSigint);

  // ブラウザが使えるかは、ここで1回だけ調べる。
  // 道具を渡すかどうかの判断に使うので、最初の発言より前に済ませておく。
  if (config.net !== false) {
    config.browserReady = await browserAvailable();
  }
  // 言語サーバーがあるかも、ここで1回だけ調べる（毎回 PATH を舐めると遅い）
  config.lspReady = anyServerAvailable();

  // 言語サーバーは子プロセスとして残る。qwc が終わるときに必ず落とす。
  // 残ると、次に起動したときに二重に立ち上がってメモリを食う。
  const cleanup = () => { try { stopLsp(); stopMcp(); } catch { /* すでに落ちている */ } };
  process.on('exit', cleanup);
  process.on('SIGINT', cleanup);
  process.on('SIGTERM', cleanup);

  /**
   * 打たれた文を、モデルへ送る形に整える。
   *
   *   /自作コマンド → その中身に差し替える
   *   @パス        → そのファイルの中身を添える
   *
   * 対話モードと -p の両方から呼ぶ。片方だけ効くと、
   * 「対話では動いたのにスクリプトでは動かない」が起きる。
   */
  function prepareInput(raw, workRoot) {
    let text = String(raw);

    if (text.startsWith('/')) {
      const [word, ...rest] = text.slice(1).split(/\s+/);
      const custom = loadCommands(workRoot).get(String(word).toLowerCase());
      if (custom) {
        info(`/${custom.name}${custom.description ? ` — ${custom.description}` : ''}`);
        text = renderCommand(custom, rest.join(' '));
      }
    }

    const mentioned = resolveMentions(text, workRoot, { vision: Boolean(config.vision) });
    for (const m of mentioned.missing) {
      warn(`@${m.name} は渡せませんでした（${m.reason}）`);
    }

    for (const a of mentioned.attachments) {
      info(`@${a.name} を添えました（${a.chars.toLocaleString()}文字${a.truncated ? '・一部' : ''}）`);
      // 丸ごと渡したものは「読んだ」ものとして扱う。
      //
      // ■ 何が起きていたか
      //   `@app.py これを書き直して` と頼むと、write_file が
      //   「まだ読んでいないファイルは丸ごと上書きさせない」で断っていた。
      //   resolveMentions は ctx を持たないので、`readFiles` に入る道が無かった
      //   （入るのは read_file・write_file・全文の受け渡しの3か所だけ）。
      //   **利用者が自分で渡したファイルを書き直せない。**
      //
      // ■ 見張りの言い分が、この経路では成り立っていない
      //   あれが守っているのは「見ていない部分が消えること」。
      //   丸ごと添えたなら、見ていない部分は無い。
      //   そのうえ buildMentionBlock は本文で
      //   「you do not need to read them again」と言っている。
      //   読むなと言っておいて、読んでいないから断る、では辻褄が合わない。
      //
      // ■ 切られた添付は入れない
      //   `truncated` が立っているなら、見えていない部分が本当にある。
      //   そこは今までどおり断るのが正しい。
      if (!a.truncated && a.path) agent.ctx.readFiles.add(a.path);
    }
    for (const img of mentioned.images) {
      info(`@${img.name} を画像として見せます（${Math.round(img.bytes / 1024)}KB）`);
    }

    return {
      text: mentioned.attachments.length ? text + buildMentionBlock(mentioned.attachments) : text,
      images: mentioned.images
    };
  }

  // ── 一回だけ実行して終わるモード ──────────────────────────
  if (opts.prompt) {
    if (!config.autoApprove) {
      // 対話できないので、確認が必要な操作は自動で断られてしまう。先に伝える。
      warn('-p では確認に答えられません。書き換えも任せるなら --yolo を付けてください。');
    }
    // 対話モードと同じ扱いにする。`@` や `/自作コマンド` が
    // -p のときだけ効かない、という差をつくらない。
    if (opts.events) {
      // 道具が実際に何を走らせたかを、モデルの文とは別に残す（local-ai-stack の verify/proofcheck が読む）
      agent.events = createEventLog(opts.events);
      agent.events.threadStarted(sessionId);
    }
    const prepared = prepareInput(opts.prompt, root);
    await agent.runTurn(prepared.text, prepared.images);
    printChanged(agent);
    process.exit(0);
  }

  // ── 対話モード ────────────────────────────────────────────
  banner(config, root);
  if (config.autoApprove) {
    // 旗で入れたのか、保存された設定で入っているのかを言い分ける。
    // 前に /save したことを忘れると「なぜ確認が出ないのか」が分からなくなり、
    // 無防備なまま気づけない。ここが唯一の手がかりになる。
    warn(
      opts.overrides.autoApprove
        ? '確認なしモードです（--yolo）。書き換えもコマンドもそのまま実行されます。'
        : '確認なしモードです（保存された設定）。書き換えもコマンドもそのまま実行されます。--confirm でこの回だけ戻せます。'
    );
  }
  // 書き換えだけ確認なしの段も、同じように言い分ける。
  // autoApprove のほうには旗と保存を区別する知らせがあるのに、こちらには
  // 何も無かった。**保存してあると、黙って書き換えが通る**ことになる。
  // 段が浅いぶん見落としやすいので、むしろこちらのほうが要る。
  else if (config.acceptEdits) {
    warn(
      opts.overrides.acceptEdits
        ? '書き換えは確認なしです（--accept-edits）。コマンドとネットは確認します。'
        : '書き換えは確認なしです（保存された設定）。コマンドとネットは確認します。/accept で戻せます。'
    );
  }
  if (config.chatMode) {
    info('雑談モードで始めました（--chat）。書き換えはしません。作業に移るときは /chat。');
  }
  // 鍵が無ければ検索の道具は渡していない。黙って使えないより、理由が分かるほうがよい。
  if (config.net !== false && !loadApiKey()) {
    warn('ネット検索は使えません（ページ取得は使えます）。' + KEY_HELP.split('\n')[0]);
  }

  for (;;) {
    // ここだけが「本人が打った依頼」。↑ で呼び出せるのはこれだけにする。
    //
    // いまどのモードにいるかは、打つ場所そのものに出す。
    // 同じタブで行き来する以上、切り替えたときの一行が流れて見えなくなったあとでも、
    // 「この一言は作業の指示として受け取られるのか」が分かる必要がある。
    const mark = config.chatMode
      ? `${c.cyan('雑談')} ${c.cyan('❯')} `
      : config.planMode
        ? `${c.brightYellow('計画')} ${c.brightYellow('❯')} `
        : `${c.magenta('❯')} `;
    const input = await ask(mark, { remember: true });
    if (input === null || input === undefined) break;
    let text = input.trim();
    if (!text) continue;

    if (text.startsWith('/')) {
      // 自分で作ったコマンドでなければ、組み込みとして処理する
      const word = text.slice(1).split(/\s+/)[0];
      if (!loadCommands(root).has(String(word).toLowerCase())) {
        const done = await handleSlash(text, { agent, config, permissions, root });
        if (done === 'exit') break;
        continue;
      }
    }

    try {
      const prepared = prepareInput(text, root);
      await agent.runTurn(prepared.text, prepared.images);

      // 計画モードで一区切りついたら、そのまま進めてよいかを聞く。
      // ここで承認を取らないと、計画モードは「書けないだけの不便なモード」で終わる。
      if (config.planMode) {
        line();
        line(`${c.brightYellow('┌')} ${c.bold('この方針で進めますか')}`);
        line(`${c.brightYellow('└')} ${c.gray('y = 実行に移る / n = このまま相談を続ける')}`);
        // そのまま Enter は「進める」でよい（この先の書き換えには改めて確認が出る）。
        // ただし**閉じた入力（Ctrl+D）は進めない**。ここは `?? ''` で受けていたので、
        // 「やっぱりやめよう」の合図で計画がそのまま実行されていた。
        // 聞き方の作法は permissions.mjs の saysYes に寄せてある。
        const raw = await ask(`${c.brightYellow('  →')} [y/n] `);
        if (saysYes(raw, { emptyMeansYes: true })) {
          config.planMode = false;
          agent.rebuildSystemPrompt();
          success('計画モードを抜けました。いまの方針で進めます。');
          await agent.runTurn(
            'その計画のとおりに実行してください。手順は変えず、終わったら何をどう変えたかを短く報告してください。'
          );
        } else {
          info('計画モードのままです。気になるところを続けてどうぞ（/plan で抜けられます）。');
        }
      }
    } catch (err) {
      line();
      error(err.message);
      if (process.env.QWC_DEBUG) line(c.gray(err.stack || ''));
    }
  }

  line();
  info('またどうぞ。');
  process.exit(0);
}

function printChanged(agent) {
  const files = agent.changedFileList();
  if (!files.length) return;
  line();
  success(`書き換えたファイル (${files.length} 件)`);
  for (const f of files) line(c.gray(`    ${path.relative(agent.root, f)}`));
}

// ── /コマンド ───────────────────────────────────────────────
async function handleSlash(text, { agent, config, permissions, root }) {
  const [cmd, ...args] = text.slice(1).split(/\s+/);
  const arg = args.join(' ').trim();

  switch (cmd) {
    case 'help':
      showSlashHelp();
      return;

    case 'exit':
    case 'quit':
      return 'exit';

    case 'clear':
      agent.clear();
      out('\x1b[2J\x1b[H');
      banner(config, root);
      info('会話をまっさらにしました。');
      return;

    case 'compact':
      await agent.compact();
      return;

    case 'model': {
      if (!arg) {
        try {
          const models = await listModels(config);
          line();
          for (const m of models) {
            line(`  ${m === config.model ? c.green('●') : c.gray('○')} ${m}`);
          }
          line();
        } catch (err) {
          error(err.message);
        }
        return;
      }
      const previous = config.model;
      config.model = arg;
      const adapted = await adaptToModel(config);
      if (adapted.canTools === false) {
        error(`${arg} はツール呼び出しに対応していないため、切り替えを取り消しました。`);
        config.model = previous;
        await adaptToModel(config);
        return;
      }
      for (const note of adapted.notes || []) {
        if (note.level === 'error') error(note.text);
        else info(note.text);
      }
      agent.rebuildSystemPrompt();
      success(`モデルを ${arg} に切り替えました。`);
      return;
    }

    case 'think': {
      // 引数なしは「いまどうなっているか」を見せるだけにする。
      // 以前はここで思考モードを入れていたが、状態を確かめる手段が無く、
      // 見るつもりで打つと勝手に切り替わっていた。
      if (arg === '') {
        await showThinkState(config);
        return;
      }

      // 深さの持ち主は config.effort ひとつ（src/effort.mjs）。
      // ここで thinkPreference を直に触ると、次の adaptToModel が effort から上書きして元に戻る。
      const wake = () => { if (config.effort === 'off') config.effort = DEFAULT_EFFORT; };
      if (arg === 'off') {
        config.effort = 'off';
      } else if (arg === 'on') {
        wake();
        config.showThinking = config.showThinking === 'off' ? 'compact' : config.showThinking;
      } else if (arg === 'full' || arg === 'compact') {
        wake();
        config.showThinking = arg;
      } else if (arg === 'quiet') {
        // 考えさせるが、様子は画面に出さない（--quiet-thinking と同じ）。
        // これまで対話中から戻す手段が無かった。
        wake();
        config.showThinking = 'off';
      } else {
        info('使い方: /think            いまの状態を見る');
        info('        /think on|off    考えさせる／考えさせない');
        info('        /think compact    考えている様子を1行だけ出す');
        info('        /think full       考えている内容を全部出す');
        info('        /think quiet      考えさせるが、様子は出さない');
        return;
      }

      // モデルが思考モードを持たない場合、ここで config.think が false に戻る。
      // だから「入れました」と言い切る前に、これを通してから状態を出す。
      const adapted = await adaptToModel(config);
      for (const note of adapted.notes || []) info(note.text);
      agent.rebuildSystemPrompt();
      await showThinkState(config);
      return;
    }

    case 'effort': {
      // 引数なしは状態を見るだけ。`/think` が「見るつもりで打つと切り替わる」で困ったので、
      // 同じ轍を踏まない。
      if (arg === '') {
        showEffortState(config);
        return;
      }
      const want = normalizeEffort(arg);
      if (want === null) {
        info('使い方: /effort              いまの深さを見る');
        info('        /effort off        考えさせない（道具を選ぶだけの作業。実測で約5倍速い）');
        info('        /effort low        浅く考える');
        info('        /effort medium     ふつう（既定）');
        info('        /effort high       深く考える（実測で6〜24倍の時間）');
        return;
      }
      config.effort = want;
      // モデルが思考モードを持たなければ、ここで config.think が false に戻る。
      // 指示文にも深さの一文が入るので、プロンプトも組み直す。
      const done = await adaptToModel(config);
      for (const note of done.notes || []) info(note.text);
      agent.rebuildSystemPrompt();
      showEffortState(config);
      return;
    }

    case 'login': {
      // 本物のブラウザを開いて、人が手でログインする。
      // qwc はキーボードにも画面の中身にも触らない。パスワードは受け取らない。
      if (!(await browserAvailable())) {
        info(BROWSER_HELP);
        return;
      }
      if (!arg) {
        info('使い方: /login github.com   （ログインしたいサイトを書いてください）');
        return;
      }
      line();
      info('ブラウザを開きます。表示された画面で、いつも通りログインしてください。');
      info('qwc はパスワードを受け取りません。終わったらここで Enter を押してください。');

      const result = await browserLogin(arg, {
        waitForUser: () => ask(`${c.brightYellow('  →')} ログインが終わったら Enter `)
      });

      if (!result.ok) {
        error(result.reason);
        return;
      }
      config.browserReady = true;
      success(`${result.host} のログイン状態を保存しました（Cookie ${result.cookieCount} 件）。`);
      info('以後 browse がこの状態でページを開きます。消すときは /logout です。');
      return;
    }

    case 'logins': {
      if (!(await browserAvailable())) {
        info(BROWSER_HELP);
        return;
      }
      const res = await savedSites();
      if (!res.ok) {
        error(res.reason);
        return;
      }
      if (!res.sites.length) {
        info('保存されているログインはありません。/login <サイト> で追加できます。');
        return;
      }
      line();
      for (const s of res.sites.slice(0, 30)) {
        line(`  ${c.bold(s.host.padEnd(34))} ${c.gray(`Cookie ${s.count} 件`)}`);
      }
      line(c.gray('\n  中身（値）は表示しません。消すには /logout。\n'));
      return;
    }

    case 'logout': {
      const res = forgetAll();
      config.browserReady = await browserAvailable();
      success(res.removed ? 'ログイン状態をすべて消しました。' : '消すものはありませんでした。');
      return;
    }

    case 'commands': {
      const found = loadCommands(root);
      if (!found.size) {
        info('自分で作ったコマンドはまだありません。');
        line(c.gray(`  .qwythos/commands/名前.md を置くと /名前 で呼べます。`));
        line(c.gray(`  中身がそのまま依頼になります（$ARGUMENTS で引数を差し込めます）。`));
        return;
      }
      line();
      for (const cmd of found.values()) {
        const clash = isReserved(cmd.name) ? c.yellow('  ← 組み込みと同じ名前なので呼べません') : '';
        line(`  ${c.cyan(`/${cmd.name}`.padEnd(16))} ${c.gray(cmd.description || '')}${clash}`);
      }
      line();
      return;
    }

    case 'plan': {
      config.planMode = !config.planMode;
      // 計画モードと雑談モードは、どちらも「いま何をする時間か」を決めるもの。
      // 両方入っていると人格と道具立ての辻褄が合わなくなるので、入った側を残す。
      if (config.planMode) config.chatMode = false;
      agent.rebuildSystemPrompt();
      if (config.planMode) {
        success('計画モードに入りました。');
        info('書き換えとコマンド実行の道具を外しました。まず調べて、方針を出します。');
      } else {
        success('計画モードを抜けました。');
      }
      return;
    }

    // 雑談モード。同じタブのまま、作業用の人格ごと入れ替える。
    // 会話そのもの（これまでのやり取り）は残したまま、いちばん上の指示文だけを差し替える。
    // 話の続きから作業に戻れるようにするためで、ここで会話を消してはいけない。
    case 'chat': {
      config.chatMode = !config.chatMode;
      if (config.chatMode) config.planMode = false;
      agent.rebuildSystemPrompt();
      if (config.chatMode) {
        success('雑談モードに入りました。');
        info('書き換えの道具を外し、指示文も雑談用に差し替えました。作業に戻るときは、もう一度 /chat。');
      } else {
        success('雑談モードを抜けました。ここからは作業モードです。');
      }
      return;
    }

    case 'todo': case 'todos': {
      const todos = agent.ctx.todos || [];
      if (!todos.length) {
        info('やることリストはまだありません。');
        return;
      }
      renderTodos(todos);
      return;
    }

    case 'yolo': {
      permissions.autoApprove = !permissions.autoApprove;
      agent.rebuildSystemPrompt();
      if (permissions.autoApprove) warn('確認なしモードにしました。書き換えもコマンドもそのまま実行されます。');
      else success('確認ありモードに戻しました。');
      return;
    }

    // 全部飛ばすのと、毎回聞かれるの間。
    // うるさいのは書き換えの確認だが、そこは差分が画面に出るので後から追える。
    // コマンドとネットは引き続き聞く（戻せないものを含み、事前には何が起きるか分からない）。
    case 'accept': {
      permissions.acceptEdits = !permissions.acceptEdits;
      agent.rebuildSystemPrompt();
      if (permissions.acceptEdits) {
        info('書き換えは確認なしで進めます。コマンド実行とネットは、これまでどおり確認します。');
        if (permissions.autoApprove) warn('いまは確認なしモード（/yolo）なので、すべて確認しません。');
      } else {
        success('書き換えも確認するように戻しました。');
      }
      return;
    }

    case 'tools': {
      // いま実際に渡している道具に印を付ける。
      // 一覧に出ているのに呼べない、という状態を作らない。
      const active = new Set(activeTools(config).map((t) => t.name));
      line();
      for (const t of TOOLS) {
        const badge = t.approval === 'never' ? c.green('読み取り') : t.approval === 'always' ? c.yellow('要確認') : c.cyan('条件つき');
        const off = active.has(t.name) ? '' : c.gray('  ← いまは渡していません');
        line(`  ${c.bold(t.name.padEnd(14))} ${badge}${off}`);
        line(c.gray(`    ${t.description.split('.')[0]}.`));
      }
      if (config.net === false) {
        line(c.gray('\n  ネット接続は切られています（--no-net）。'));
      } else if (!loadApiKey()) {
        line(c.gray(`\n  ${KEY_HELP.split('\n')[0]}`));
      }
      line();
      return;
    }

    case 'stats': {
      const s = agent.stats;
      const secs = (ms) => `${((ms || 0) / 1000).toFixed(1)} 秒`;
      line();
      line(`  ${c.gray('やり取り')}      ${s.turns} 回`);
      line(`  ${c.gray('ツール実行')}    ${s.toolCalls} 回`);
      line(`  ${c.gray('入力トークン')}  ${s.inputTokens.toLocaleString()}`);
      line(`  ${c.gray('出力トークン')}  ${s.outputTokens.toLocaleString()}`);
      line(`  ${c.gray('会話の長さ')}    ${agent.messages.length} 件`);
      // トークン数と、その長さでの速度。実測値は src/ctxcost.mjs。
      line(`  ${c.gray('文脈')}          ${contextLine(agent.contextTokens(), config.numCtx)}`);
      // 同じファイルを読み直したときに、二度積まずに済んだ量。
      // どれだけ起きるかは記録から推定できなかった（保存されるのは圧縮後の履歴のため）ので、
      // 実際に使いながら数える。0 が続くなら、この機構は要らない。
      if (s.dedupedChars) {
        line(`  ${c.gray('読み直しの重複')}  ${s.dedupedChars.toLocaleString()} 字を省いた ${
          c.gray(`（約 ${Math.round(s.dedupedChars / 3.5).toLocaleString()} トークン）`)}`);
      }
      // 待ち時間の内訳。
      //
      // 「今日はなぜか遅い」の中身は、たいてい生成ではなく前処理と読み込みにある。
      // 合計だけ見せても切り分けられないので、3つに割って出す。
      if (s.totalMs) {
        line();
        line(`  ${c.gray('モデルを待った時間')}  ${secs(s.totalMs)}`);
        line(`    ${c.gray('うち読み込み')}      ${secs(s.loadMs)}`);
        line(`    ${c.gray('うち前処理')}        ${secs(s.promptMs)} ${c.gray('（送った会話を読む時間）')}`);
        line(`    ${c.gray('うち生成')}          ${secs(s.evalMs)}${
          s.evalMs ? c.gray(`（${(s.outputTokens / (s.evalMs / 1000)).toFixed(1)} tok/s）`) : ''
        }`);
      }
      line();
      return;
    }

    case 'files':
      printChanged(agent);
      if (!agent.changedFileList().length) info('まだ何も書き換えていません。');
      return;

    case 'diff': {
      // 途中の経過ではなく、始める前と今の違いを出す。
      // 3回直したファイルを3回ぶん見せられても、結局どうなったのかは分からない。
      const target = arg ? path.resolve(root, arg) : null;
      const changes = sessionChanges(agent.ctx, target).filter((ch) => ch.changed);
      if (!changes.length) {
        info(arg ? `${arg} は、このセッションでは書き換えていません。` : 'このセッションでの書き換えはまだありません。');
        return;
      }
      for (const ch of changes) {
        const rel = path.relative(root, ch.path) || ch.path;
        line();
        if (ch.unreadable) {
          line(`${c.brightCyan('●')} ${c.bold(rel)} ${c.gray('— いま読めません')}`);
          continue;
        }
        if (ch.big) {
          line(`${c.brightCyan('●')} ${c.bold(rel)} ${c.gray('— 大きすぎるか、読めない形式のため差分は出せません')}`);
          continue;
        }
        if (ch.removed) {
          line(`${c.brightCyan('●')} ${c.bold(rel)} ${c.red('— 削除されています')}`);
          continue;
        }
        const label = ch.created ? c.green('新規') : c.gray('変更');
        line(`${c.brightCyan('●')} ${c.bold(rel)} ${label}`);
        line(renderDiff(ch.before ?? '', ch.after ?? ''));
      }
      line();
      return;
    }

    case 'undo': {
      // 戻すのは「直近の1手」ではなく「直近のお願い」ぶん。
      // 1回のお願いで3ファイル直したうちの1つだけ戻しても、半端な状態が残る。
      if (!canUndo(agent.ctx)) {
        info('戻せる書き換えがありません（このセッションで書き換えたぶんだけ戻せます）。');
        return;
      }
      const result = undoLastTurn(agent.ctx);
      if (!result) {
        info('戻せる書き換えがありません。');
        return;
      }
      for (const r of result.restored) {
        const rel = path.relative(root, r.path) || r.path;
        success(r.removed ? `${rel} を消しました（新しく作られたファイルです）` : `${rel} を元に戻しました`);
      }
      for (const sk of result.skipped) {
        const rel = path.relative(root, sk.path) || sk.path;
        warn(`${rel} はそのままにしました — ${sk.reason}`);
      }
      // 控えの上限に当たって捨てたぶんがあれば、必ず言う。
      // 黙っていると「元に戻しました」だけが画面に残り、実際には書き換えが残る。
      if (result.dropped) {
        warn(
          `このお願いは書き換えが多く、控えの上限（${MAX_ENTRIES} 件）を超えました。` +
          `早いほうの ${result.dropped} 件は控えが残っておらず、戻せていません。` +
          'そのぶんの書き換えはファイルに残ったままです（/diff で見られます）。'
        );
      }
      if (!result.restored.length && !result.skipped.length) info('戻すものがありませんでした。');
      // モデルにも伝える。黙って戻すと、直したつもりのまま次の一手を組み立てる
      if (result.restored.length) {
        const names = result.restored.map((r) => path.relative(root, r.path) || r.path).join(', ');
        agent.messages.push({
          role: 'user',
          content:
            `[The user undid your last set of file changes. These files are back to their previous ` +
            `content: ${names}. Do not assume your edits are still there. Re-read a file before ` +
            `changing it again, and ask what to do differently instead of repeating the same edit.]`
        });
      }
      return;
    }

    case 'refine': {
      const harness = loadHarness(root);

      if (arg === 'list') {
        line();
        const show = (notes, label) => {
          if (!notes.length) return;
          line(`  ${c.bold(label)}`);
          for (const n of notes) {
            line(`    ${c.gray(n.id)} ${n.text}`);
            line(`         ${c.gray(`根拠: ${n.evidence}`)}`);
          }
        };
        show(harness.project, 'このフォルダ');
        show(harness.global, '全体');
        if (!harness.project.length && !harness.global.length) info(describeHarness(harness));
        line();
        return;
      }

      if (arg === 'undo') {
        const undone = undoHarness(root);
        if (!undone.length) info('取り消せる変更がありません。');
        else {
          agent.rebuildSystemPrompt();
          success('直前の見直しを取り消しました。');
        }
        return;
      }

      const spinner = new Spinner('やり取りを見直しています').start();
      const result = await agent.refine(arg);
      spinner.stop();

      if (!result.applied.length) {
        info(result.reason || '変更はありませんでした。');
        return;
      }
      line();
      for (const a of result.applied) {
        const mark = a.op === 'delete' ? c.red('−') : a.op === 'update' ? c.cyan('~') : c.green('+');
        line(`  ${mark} ${a.scope === 'global' ? c.gray('[全体]') : c.gray('[このフォルダ]')} ${a.text}`);
      }
      line();
      info('取り消すなら /refine undo。一覧は /refine list。');
      return;
    }

    case 'save': {
      saveConfig({
        model: config.model,
        host: config.host,
        numCtx: config.numCtx,
        temperature: config.temperature,
        // 深さの持ち主は effort ただ1つ（src/effort.mjs）。**ここに無かった。**
        // think だけを保存しても、次の起動では DEFAULT_CONFIG の effort:'medium' が
        // 必ず入っているので、ollama.mjs の逃げ道（effort が undefined のときだけ think を見る）に
        // 一度も入らない。実測: `/effort off` して `/save` しても、次の起動で medium に戻る。
        effort: config.effort,
        // 古い qwc が読めるように、think も併せて残す（読む側は config.mjs が effort へ寄せる）
        think: config.thinkPreference ?? config.think,
        showThinking: config.showThinking,
        maxSteps: config.maxSteps,
        // 確認をどこまで飛ばすかは好みが分かれるので、次回にも持ち越せるようにする。
        // 全部飛ばす設定（autoApprove）も保存する ── 2026-08-25 に本人が選択。
        // 「一度うっかり入れたまま忘れると、以後ずっと無防備になる」危険は残るので、
        // 起動のたびに保存された設定だと警告し、その回だけ戻せる --confirm を用意してある。
        acceptEdits: config.acceptEdits,
        autoApprove: config.autoApprove
      });
      success('いまの設定を次回以降の既定にしました。');
      if (config.autoApprove) {
        warn('確認なしモードも保存しました。次からも確認せずに動きます（--confirm でその回だけ戻せます）。');
      }
      return;
    }

    case 'init': {
      await agent.runTurn(
        'Explore this project and write a QWYTHOS.md at the workspace root. ' +
        'It must cover: what the project is, how to run it, how to test it, the folder layout, ' +
        'and any conventions a new contributor must follow. Base every line on files you actually read. ' +
        'Keep it under 100 lines.'
      );
      return;
    }

    default:
      warn(`知らないコマンドです: /${cmd}（/help で一覧）`);
  }
}

main().catch((err) => {
  error(err.message);
  if (process.env.QWC_DEBUG) line(c.gray(err.stack || ''));
  process.exit(1);
});
