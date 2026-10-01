/**
 * events.mjs — 走った記録を `codex exec --json` と同じ形で残す。
 *
 * ■ 何のためにあるのか
 *   「テストを実行しました」が本当かどうかは、モデルの文からは分からない。
 *   道具を実際に動かしたのは qwc なので、qwc が「何を走らせ、どう終わったか」を
 *   その場で書き残せば、文と突き合わせられる（verify/proofcheck --codex-jsonl）。
 *
 * ■ なぜ Codex の形なのか
 *   OpenAI が openai/codex で公開しているハーネスの出力形式に合わせておけば、
 *   Codex を走らせた記録と qwc を走らせた記録を、同じ道具で同じように読める。
 *   形は codex-rs/exec/src/exec_events.rs の ThreadEvent / ThreadItem に従う。
 *   qwc が持っていない種類（推論の要約・MCP・Web 検索など）は出さない。
 *
 * ■ 置き場所の注意
 *   この記録はモデルの手が届く場所に置かない。作業場の中に置くと、
 *   run_command で書き換えられてしまい、証拠にならない。
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

export function createEventLog(file) {
  const abs = path.resolve(file);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  // 前の記録に続けて書くと、別の走りの命令が混ざる。毎回まっさらにする。
  fs.writeFileSync(abs, '');
  let seq = 0;

  // 1行ずつ同期で書く。途中で落ちても、そこまでの記録は残る。
  const write = (event) => fs.appendFileSync(abs, JSON.stringify(event) + '\n');
  const nextId = () => `item_${seq++}`;

  return {
    file: abs,
    threadStarted(threadId = crypto.randomUUID()) {
      write({ type: 'thread.started', thread_id: threadId });
    },
    turnStarted() {
      write({ type: 'turn.started' });
    },
    /** 道具が返した event をそのまま 1 件の item として残す */
    item(details) {
      write({ type: 'item.completed', item: { id: nextId(), ...details } });
    },
    agentMessage(text) {
      write({ type: 'item.completed', item: { id: nextId(), type: 'agent_message', text: String(text ?? '') } });
    },
    turnCompleted({ inputTokens = 0, outputTokens = 0 } = {}) {
      write({
        type: 'turn.completed',
        usage: {
          input_tokens: inputTokens,
          cached_input_tokens: 0,
          output_tokens: outputTokens,
          reasoning_output_tokens: 0
        }
      });
    },
    turnFailed(message) {
      write({ type: 'turn.failed', error: { message: String(message) } });
    }
  };
}
