// 設定の読み書き。~/.qwythos-code/config.json があればそれを既定値に上書きする。
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

export const HOME_DIR = path.join(os.homedir(), '.qwythos-code');
export const CONFIG_PATH = path.join(HOME_DIR, 'config.json');
export const SESSION_DIR = path.join(HOME_DIR, 'sessions');

export const DEFAULT_CONFIG = {
  // 接続先
  host: 'http://localhost:11434',
  // 既定は gemma4:26b（2026-08-07 に qwythos:latest から変更）。
  //
  // 26B だが MoE で実効3.8Bなので、9B より速い。実測では速さと正しさの両方で上回った：
  // 同じ課題（既存の関数を使い回して新しい関数を足す）で
  //   9B      = 46.5秒・引数を取り違えた壊れたコード。存在しないパスも3手試した
  //   gemma4  = 36秒（18GBの読み込み込み）・正しいコード。自分からやることリストも作った
  // 14B は論外（同じ作業に20倍かかる。5手の修正で 30秒 → 10分超）。
  //
  // 必要なのは 18GB。GPU枠に載らない環境では `-m qwythos:latest`（7.4GB）に落とす。
  model: 'gemma4:26b',

  // 既定のモデルが GPU に載りきらなかったときの逃げ先（7.4GB）。
  //
  // 載りきらないと、はみ出した分が CPU 側で動いて極端に遅くなる。
  // そうなるくらいなら、遅いが GPU に収まるほうを使う。
  // 切り替えるのは載らなかったときだけで、作業の内容では切り替えない
  // （実測では 26B のほうが 9B より速く、正しい。落とす理由が容量以外に無い）。
  lightModel: 'qwythos:latest',
  autoDowngrade: true,

  // 生成パラメータ（qwythos の Modelfile に合わせた値）
  numCtx: 32768,
  temperature: 0.3,
  topP: 0.95,
  topK: 20,
  repeatPenalty: 1.05,
  keepAlive: '30m',
  think: true,

  // 考える深さ（off / low / medium / high）。/effort で変える。
  // 深さの意味と実測値は src/effort.mjs にまとめてある。
  effort: 'medium',

  // モデルの返事をどれだけ待つか。
  //
  // モデルの読み込みや文脈枠の取り直しが挟まると、最初の1文字までが分単位になる。
  // Node の fetch はここを300秒で必ず打ち切るので、この2か所だけ node:http で投げている。
  firstTokenMs: 15 * 60 * 1000,

  // 出はじめたあとで途切れたときに見切る長さ。
  //
  // 3分では短すぎた。**Ollama は道具の呼び出しを、書き終えるまで送ってこない。**
  // 生成しているあいだ、こちらには1バイトも届かない（実測で、道具を1つ返すだけの
  // やり取りに14秒の無音があった）。長い引数——大きなファイルの書き込みなど——を
  // 組み立てているときは、その無音がそのまま生成時間になる。
  // 35 tok/s なら3分＝6,300トークンで、普通に届く長さ。
  //
  // 実際、正常に終わったやり取りが Ollama 側では 1m22s〜4m46s かかっていた。
  // つまり3分の見切りは、**動いている作業を殺していた**。
  // 10分＝約21,000トークンぶんの生成にあたり、まともな1回の道具呼び出しでは届かない。
  stallMs: 10 * 60 * 1000,

  // 依頼ごとモデルを降ろされたとき、何秒待って掛け直すか（段数＝掛け直す回数）。
  //
  // Ollama は処理中でもモデルを降ろす。同じモデルを別の広さ（num_ctx）で呼ばれても、
  // 別のモデルに GPU の枠を取られても起きる。降ろされた側に返るのは
  // HTTP 500 "unexpected EOF" だけで、そこまでの前処理は丸ごと消える。
  // 2026-08-31 の朝、60秒おきの点検がこれを起こし続け、qwc は1手も終われなかった。
  // 空配列にすれば掛け直さない（試験で使う）。
  retryWaitsMs: [3000, 10000, 30000],

  showThinking: 'compact', // compact（1行だけ流す）/ full（全部出す）/ off（出さない）

  // 考えた内容を、その手の道具を使い終わった時点で捨てるか。
  //
  // 1回のお願いは最大200手まで回るので、残したままだと過去の思考が全部積み上がり、
  // 手が進むたびに送り直すことになる。道具の呼び出しと結果が残っていれば何をしたかは辿れる。
  // false にすると、1回のお願いの中では思考を残したままにする（比べたいときに使う）。
  dropThinkingAfterTools: true,

  // 前の依頼で読んだ道具の出力を、次の依頼が来た時点で短くするか。
  //
  // 文脈を太らせているのは、ほぼこれだけ（実測: 6件頼んだ会話で 0 → 68,247 文字。
  // 指示文は一定、考えは0、本文は 638 文字）。太ると1手の前処理が 0.9秒 → 53.4秒、
  // 生成が 37.0 → 21.7 tok/s まで落ちる。
  // `compactAtRatio`(0.7) の圧縮は 45,875 トークンを超えるまで働かないので、
  // ふつうの作業では最後まで発動しない。
  shrinkOldToolOutput: true,
  keepFullToolTurns: 1,      // 直前の依頼のぶんは丸ごと残す
  oldToolOutputChars: 400,   // それより古いものは、頭のこれだけ残して切る

  // 同じファイルを読み直したとき、まったく同じ中身を二度積まない。
  //
  // 実測（198セッション）: 間に編集を挟まない読み直しが 149,292 字あり、
  // 全文脈の 10.4% を占めていた。同じセッションで ChatView.tsx を9回、
  // tools.mjs を8回読んでいる。1回 12,191 字のものが丸ごと積み上がる。
  //
  // 判定は「出力が一字一句同じで、その写しがまだ履歴に残っている」だけで行う。
  // 中身が変わっていれば文字列が変わるので、変更の検知は自動で付いてくる。
  // 会話が長くなって生成が目に見えて遅くなったら、一度だけ知らせる。
  // 知らせるだけで、こちらからは何もしない（切るかどうかは持ち主が決める）。
  contextNotice: true,

  dedupeToolOutput: true,
  dedupeMinChars: 800,       // これより短い出力は、置き換えても得が無いので触らない

  // 応答ごとに、かかった時間の内訳を1行出すか。
  //
  // ローカルでは待ち時間の大半が生成ではなく、モデルの読み込みと前処理で消える。
  // 内訳が見えないと「今日はなぜか遅い」で終わり、文脈を広げすぎたのか、
  // モデルが GPU に載り切っていないのかを切り分けられない。
  // 2秒未満の応答には出さない（速いときに毎回付くと、ただの雑音になる）。
  showTiming: true,

  // 書き換えだけ確認なしで進めるか（`--accept-edits` / `/accept`）。
  //
  // 全部飛ばす（`--yolo`）と、毎回聞かれるの間。
  // ふだんうるさいのは書き換えの確認で、そこは差分が画面に出るので後から追える。
  // コマンド実行とネットは引き続き聞く（戻せないものを含み、事前には何が起きるか分からない）。
  acceptEdits: false,

  // 書き換えた中身を画面に出すか。
  //
  // 「1 か所を置き換え」だけでは、何がどうなったのか分からない。
  // 確認を出さない設定（--yolo）では、ここが唯一の手がかりになる。
  // 確認欄で既に見せたときは重ねて出さない。
  showDiff: true,

  // エージェントの動作
  // 1回のお願いで許すツール実行の往復上限。
  //
  // 40 では実作業で足りない。実測で普通の修正が16手、テストを書いて回すと
  // その倍は使うので、少し込み入った依頼だと途中で壁に当たって
  // 「続けて」と打ち直すことになっていた。
  //
  // 上げても文脈は破綻しない。毎手 maybeCompact() が走り、
  // numCtx の 70%（compactAtRatio）を超えたら古いツール出力を短くし、
  // それで足りなければ要約して詰める。
  //
  // 暴走の歯止めは回数ではなく別の3つが担っている：
  //   - 同じツール呼び出しの繰り返し（duplicateLimit: 3）
  //   - 何も変えずに喋るだけのとき（maxNudges: 5）
  //   - Ctrl+C（ctx.signal で実行中のコマンドごと止まる）
  // なので、ここは「本当に何かがおかしいときの最後の網」として大きく取る。
  maxSteps: 200,
  // 道具の出力を、**履歴に入れる瞬間に**この長さで切る（freeze-on-write）。
  //
  // 12,000 から 4,000 に下げた（2026-09-03）。切るのは追記時の一度だけで、
  // **以後この文字列には二度と触らない。** 後から書き換えると llama.cpp の
  // prompt cache が先頭一致を失い、Gemma 4 では 106MiB の checkpoint が
  // 死んだまま枠を占有する（agent.mjs の runTurn の注記を参照）。
  //
  // 4,000 にした根拠: 12,000 で実測したとき 1手の前処理が 12〜27秒だったのに対し、
  // 4,000 では 0.3〜13秒で、キャッシュ命中（315ms / 378ms）も観測できた。
  // 切った箇所には read_file の offset/limit で読み直せることを書いてある（truncateOutput）。
  maxToolChars: 4000,
  maxFileBytes: 400000,   // これより大きいファイルは丸ごと読まない
  commandTimeoutMs: 120000,
  compactAtRatio: 0.7,    // 文脈がこの割合を超えたら自動で要約圧縮
  duplicateLimit: 3,      // 同じツール呼び出しが続いたら止める回数
  maxNudges: 5,           // 「やります」と言うだけで動かないときに促す回数の上限
  todoHintAfter: 6,       // これだけ道具を使ってもリストが無ければ、1度だけ促す
  // これだけ道具を使っても何も変わっていなければ、1度だけ区切りを促す。
  // 「どこを直すか書いていない依頼」で延々と読み続けるのを止めるため
  // （実測: 28ファイルの案件で7分ぶん読み続け、同じファイルを3回読み直して何も変えなかった）。
  exploreLimit: 10,

  // ネット接続（web_search / web_fetch）
  //
  // 手元のファイルを触るのとは性質が違う。検索の言葉と URL が外へ出るので、
  // 既定では毎回確認する（y/n/a）。丸ごと切りたいときは net: false。
  net: true,
  netTimeoutMs: 20000,
  maxFetchBytes: 2000000,  // 1ページで受け取る上限。文脈を食いつぶさないため

  // ログイン済みブラウザ（browse）。Playwright が入っているときだけ有効になる。
  // 判定は起動時に1回だけ行い、browserReady に入れる（毎回試すと遅い）。
  browserReady: false,
  browseTimeoutMs: 30000,

  // 言語サーバー（find_symbol）が使えるか。起動時に1回だけ調べて入れる。
  lspReady: false,

  // 計画モード。true のあいだは書き換える道具を渡さない。
  // 保存はしない（次回まで持ち越すものではないため）。
  planMode: false,

  // 雑談モード。true のあいだは指示文を雑談用に差し替え、書き換える道具を渡さない。
  // 計画モードと同じく保存はしない。次に起動したときは作業モードで始まる。
  chatMode: false,

  // 安全まわり
  autoApprove: false,      // true で確認なし（--yolo と同じ）
  allowOutsideRoot: false, // 作業フォルダの外を触れるか
  safeCommands: [
    'ls', 'pwd', 'cat', 'head', 'tail', 'wc', 'which', 'echo', 'date',
    'file', 'stat', 'find', 'grep', 'rg', 'fd', 'tree', 'du', 'df',
    'node -v', 'node --version', 'npm ls', 'python3 -V', 'python3 --version',
    'git status', 'git diff', 'git log', 'git branch', 'git show', 'git remote'
  ]
};

function ensureHomeDir() {
  fs.mkdirSync(HOME_DIR, { recursive: true });
  // 会話の記録は本人だけが読めるようにする（理由は session.mjs に）
  fs.mkdirSync(SESSION_DIR, { recursive: true, mode: 0o700 });
  try { fs.chmodSync(SESSION_DIR, 0o700); } catch { /* 締められなくても止めない */ }
}

/**
 * 数で入っているべき設定のうち、**壊れていたら既定に戻す**もの。
 *
 * `--ctx 32k` のように数にならない値を渡すと NaN になり、`/save` すると JSON には
 * `null` が残る。`null` は既定値を上書きするので、以後ずっと居座る。
 * numCtx が壊れると文脈のしきい値も壊れ、毎ターン要約が走る（bin/qwc.mjs の asNumber を参照）。
 * 旗のほうは通す前に止めるようにしたが、**すでに壊れて保存されているものは、ここで拾う**。
 */
const NUMERIC_KEYS = [
  'numCtx', 'temperature', 'topP', 'topK', 'repeatPenalty',
  'maxSteps', 'maxToolChars', 'maxFileBytes', 'commandTimeoutMs', 'compactAtRatio',
  'duplicateLimit', 'maxNudges', 'todoHintAfter', 'exploreLimit',
  'netTimeoutMs', 'maxFetchBytes', 'browseTimeoutMs', 'firstTokenMs', 'stallMs',
  'dedupeMinChars', 'keepFullToolTurns', 'oldToolOutputChars'
];

/**
 * 保存された設定を、使える形に直してから既定に重ねる。
 *
 * 直すのは2つだけ。どちらも「黙って居座る」たぐいの壊れ方をする。
 *   ・数として読めない値  … `--ctx 32k` を `/save` すると null が残り、既定を上書きし続ける
 *   ・古い `think` だけの形 … 深さの持ち主は effort なので、そちらへ移す
 */
export function normalizeStoredConfig(fromFile, { warn = (m) => process.stderr.write(m) } = {}) {
  const out = dropBrokenNumbers(fromFile, warn);
  // 古い設定ファイルには `think` しか入っていない。
  //
  // ollama.mjs にも逃げ道はあるが、そこは `cfg.effort === undefined` を条件にしていて、
  // DEFAULT_CONFIG が必ず effort を入れる以上**一度も通らない**。だから読む側で直す。
  if (out.effort === undefined && out.think !== undefined) {
    out.effort = out.think === false ? 'off' : DEFAULT_CONFIG.effort;
  }
  return out;
}

/**
 * **0 に意味がある鍵。**
 *
 * 以前はこの一覧の全部で「0 は壊れている」扱いにしていた。理由は
 * 「既定値が全部 0 より大きいから」だったが、**既定が正なことは 0 が無効な理由にならない。**
 *   temperature: 0        … 毎回同じ答えを出す（計測の再現に要る）
 *   topK: 0               … 無効化（llama.cpp の約束）
 *   maxNudges: 0          … 促しを出さない
 *   keepFullToolTurns: 0  … 道具の出力を1つも丸ごと残さない
 *   oldToolOutputChars: 0 … 古い道具の出力を残さない
 * どれも「切る」という指定で、0 でしか書けない。
 * 0 を落として既定に戻すと、**切ったつもりが効いていない**という一番たちの悪い形になる。
 * （別セッション daigo-de が maxNudges: 0 で見つけた・2026-09-26）
 *
 * 時間・大きさ・回数の上限（commandTimeoutMs / maxSteps / compactAtRatio など）は
 * 0 で動かなくなるので、引き続き 0 も壊れている扱いにする。
 */
const ZERO_OK_KEYS = new Set([
  'temperature', 'topK', 'maxNudges', 'keepFullToolTurns', 'oldToolOutputChars',
]);

function dropBrokenNumbers(source, warn) {
  const out = { ...source };
  for (const key of NUMERIC_KEYS) {
    if (!(key in out)) continue;
    const v = Number(out[key]);
    const 下限 = ZERO_OK_KEYS.has(key) ? 0 : 1;
    if (Number.isFinite(v) && v >= 下限) continue;
    // **理由を正しく言う。** 0 は数として読める。読めないのと、小さすぎるのは別の話。
    const 理由 = Number.isFinite(v)
      ? `設定の ${key} は ${下限 === 0 ? '0 以上' : '0 より大きい数'} にしてください（${JSON.stringify(out[key])}）。`
      : `設定の ${key} が数として読めません（${JSON.stringify(out[key])}）。`;
    warn(`${理由}既定値 ${DEFAULT_CONFIG[key]} を使います: ${CONFIG_PATH}\n`);
    delete out[key];
  }
  return out;
}

export function loadConfig() {
  ensureHomeDir();
  let fromFile = {};
  if (fs.existsSync(CONFIG_PATH)) {
    try {
      fromFile = normalizeStoredConfig(JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8')));
    } catch (err) {
      process.stderr.write(`設定ファイルが読めませんでした (${CONFIG_PATH}): ${err.message}\n`);
    }
  }
  const env = {};
  if (process.env.OLLAMA_HOST) {
    const h = process.env.OLLAMA_HOST;
    env.host = h.startsWith('http') ? h : `http://${h}`;
  }
  if (process.env.QWC_MODEL) env.model = process.env.QWC_MODEL;
  return { ...DEFAULT_CONFIG, ...fromFile, ...env };
}

export function saveConfig(patch) {
  ensureHomeDir();
  let current = {};
  if (fs.existsSync(CONFIG_PATH)) {
    try {
      current = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));
    } catch {
      current = {};
    }
  }
  const next = { ...current, ...patch };
  fs.writeFileSync(CONFIG_PATH, `${JSON.stringify(next, null, 2)}\n`, 'utf8');
  return next;
}
