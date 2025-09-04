import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import OpenAI from "openai";

/**
 * SODERBOT for Render (safe HTML join, mobile-friendly)
 * - Binds to process.env.PORT
 * - Crawls EN pages from sodermanaudiovisual.com
 * - Hybrid retrieval: OpenAI embeddings + TF-IDF sparse scoring
 * - Answers in EN/FI/SV (output language) using EN content
 * - /health, /kb-status, /reindex
 * - Fancy floating UI with iPhone-safe bottom sheet
 */

const app = express();
const PORT = Number(process.env.PORT) || 3000;
const OPENAI_KEY = process.env.OPENAI_API_KEY;

// Fail fast if missing key
if (!OPENAI_KEY || OPENAI_KEY.startsWith("sk-your_")) {
  console.error("❌ Missing OPENAI_API_KEY. Set it in Render → Service → Environment.");
  process.exit(1);
}

const openai = new OpenAI({ apiKey: OPENAI_KEY });

const SITE = "https://www.sodermanaudiovisual.com";
const MAX_PAGES = 120;
const CHUNK_SIZE = 800;
const MIN_DOC_CHARS = 180;
const MAX_CONTEXT = 9000;
const TOP_K = 12;
const BATCH_EMBED = 96;

app.use(cors());
app.use(bodyParser.json());

// In-memory KB + sparse index
let KB = [];                         // { id, url, lang:'en', chunk, vec? }
let VOCAB_IDF = new Map();           // term -> idf
let CHUNK_TF = new Map();            // id -> Map(term -> tf)

/* ---------------- Utils ---------------- */
function chunkText(s, n = CHUNK_SIZE) {
  const out = []; for (let i = 0; i < s.length; i += n) out.push(s.slice(i, i + n));
  return out;
}
function stripHtml(html) {
  if (!html) return "";
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<!--[\s\S]*?-->/g, " ")
    .replace(/<\/(p|div|section|article|li|h1|h2|h3|br)>/gi, "\n")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}
function tokenize(s) {
  return String(s).toLowerCase().split(/[^a-z0-9äöåæø]+/i).filter(w => w && w.length > 1);
}
function unique(arr) { return Array.from(new Set(arr)); }
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { const x = a[i], y = b[i]; dot += x * y; na += x * x; nb += y * y; }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}
function escapeForWordBoundary(term) {
  return term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/* ---------------- URL discovery ---------------- */
async function getSitemapUrls() {
  const set = new Set();
  try {
    const r = await fetch(SITE + "/sitemap.xml", { headers: { "User-Agent": "Mozilla/5.0 Chrome/127 Safari/537.36" } });
    if (r.ok) {
      const xml = await r.text();
      const re = /<loc>([^<]+)<\/loc>/gi;
      let m;
      while ((m = re.exec(xml))) {
        const u = m[1].trim().replace(/\/+$/, "");
        const path = new URL(u).pathname.replace(/^\/+/, "");
        if (path.startsWith("fi") || path === "fi" || path.startsWith("sv") || path === "sv") continue; // EN only
        if (u.startsWith(SITE)) set.add(u);
      }
    }
  } catch {}
  ["/", "/home", "/services", "/about-us", "/contact", "/references", "/get-a-quote", "/consultation"].forEach(p => {
    try { set.add(new URL(p, SITE).toString().replace(/\/+$/, "")); } catch {}
  });
  return Array.from(set).slice(0, MAX_PAGES);
}

/* ---------------- Page fetch ---------------- */
async function fetchPageStrong(url) {
  try {
    const r = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml"
      }
    });
    if (!r.ok) return null;
    const html = await r.text();
    const title = (html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1] || "").trim();
    const meta = (html.match(/<meta[^>]+name=["']description["'][^>]+content=["']([^"']+)["']/i)?.[1] || "").trim();
    let body = "";
    const main = html.match(/<main[\s\S]*?<\/main>/i); if (main) body += " " + stripHtml(main[0]);
    const art = html.match(/<article[\s\S]*?<\/article>/i); if (art) body += " " + stripHtml(art[0]);
    if (body.trim().length < 120) body = stripHtml(html);
    const doc = [title ? ("TITLE: " + title) : "", meta ? ("DESCRIPTION: " + meta) : "", body ? ("BODY:\n" + body) : ""]
      .filter(Boolean).join("\n\n");
    return (doc && doc.length >= MIN_DOC_CHARS) ? doc : null;
  } catch { return null; }
}

/* ---------------- Crawl + index + embed ---------------- */
async function crawl() {
  KB = []; VOCAB_IDF.clear(); CHUNK_TF.clear();

  const urls = await getSitemapUrls();
  let idCounter = 0;

  for (const url of urls) {
    const doc = await fetchPageStrong(url);
    if (!doc) { console.log("· skip", url); continue; }
    const pieces = chunkText(doc);
    for (const c of pieces) {
      KB.push({ id: idCounter++, url, lang: "en", chunk: c });
    }
    console.log("✓ [en]", url, "chunks:", pieces.length, "chars:", doc.length);
  }
  console.log("📚 Crawled chunks:", KB.length);

  // Sparse index
  const N = KB.length || 1;
  const df = new Map();
  for (const d of KB) {
    const toks = tokenize(d.chunk);
    const terms = unique(toks);
    const tf = new Map();
    for (const t of terms) {
      const re = new RegExp("\\b" + escapeForWordBoundary(t) + "\\b", "gi");
      const count = (d.chunk.match(re) || []).length || 1;
      tf.set(t, count);
    }
    CHUNK_TF.set(d.id, tf);
    for (const t of terms) df.set(t, (df.get(t) || 0) + 1);
  }
  for (const [t, dfVal] of df.entries()) {
    const idf = Math.log((N + 1) / (dfVal + 0.5));
    VOCAB_IDF.set(t, idf);
  }
  console.log("🔎 Sparse index ready. Terms:", VOCAB_IDF.size);

  await embedAllChunks();
}

async function embedAllChunks() {
  const texts = KB.map(d => d.chunk);
  let start = 0;
  const dims = new Set();
  while (start < texts.length) {
    const batch = texts.slice(start, start + BATCH_EMBED);
    const resp = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: batch
    });
    resp.data.forEach((e, i) => {
      KB[start + i].vec = e.embedding;
      dims.add(e.embedding.length);
    });
    start += BATCH_EMBED;
  }
  console.log("🧠 Embeddings ready for", KB.length, "chunks. Dim:", Array.from(dims).join(","));
}

/* ---------------- Retrieval (hybrid) ---------------- */
async function retrieveContext(query) {
  if (!KB.length) return "";
  const emb = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: query
  });
  const qv = emb.data[0].embedding;

  const terms = unique(tokenize(query));
  const scored = [];
  for (const d of KB) {
    const dense = d.vec ? cosine(qv, d.vec) : 0;
    let sparse = 0;
    const tf = CHUNK_TF.get(d.id);
    if (tf) {
      for (const t of terms) {
        const f = tf.get(t) || 0;
        const idf = VOCAB_IDF.get(t) || 0;
        const denom = 1 + 0.25 + f; // simplified BM25-ish
        sparse += (idf * (f * 1.2) / denom);
      }
    }
    const score = 0.7 * dense + 0.3 * Math.tanh(sparse);
    if (score > 0) scored.push({ d, score });
  }
  scored.sort((a, b) => b.score - a.score);
  let ctx = "";
  for (const { d } of scored.slice(0, TOP_K)) {
    if (ctx.length + d.chunk.length > MAX_CONTEXT) break;
    ctx += "[Source] " + d.url + "\n" + d.chunk + "\n\n";
  }
  return ctx;
}

/* ---------------- Chat API ---------------- */
app.post("/chat", async (req, res) => {
  try {
    const msg = req.body?.message || "";
    const lang = req.body?.lang || "en";
    if (!msg) return res.json({ reply: "Please type a message." });

    const ctx = await retrieveContext(msg);
    const langRule =
      (lang === "fi") ? "Answer in Finnish. Be faithful to the facts from the knowledge."
      : (lang === "sv") ? "Answer in Swedish. Be faithful to the facts from the knowledge."
      : "Answer in English. Be faithful to the facts from the knowledge.";

    const sys = [
      "You are SODERBOT, assistant for Soderman Audiovisual.",
      "We are a film production company based in Vaasa and Helsinki. Use 'we'/'our'.",
      "When the question is very short (even a single word), infer the most relevant section from the knowledge and explain briefly with context.",
      langRule,
      "Cite implicit context only from the provided knowledge. If something is missing, say so briefly and offer a human handoff.",
      ctx ? ("Knowledge:\n" + ctx) : "Knowledge: (none yet)"
    ].join("\n");

    const r = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.15,
      messages: [
        { role: "system", content: sys },
        { role: "user", content: msg }
      ]
    });

    res.json({ reply: r.choices?.[0]?.message?.content || "(no reply)" });
  } catch (e) {
    res.json({ reply: "Error " + (e?.message || String(e)) });
  }
});

/* ---------------- Health / Status / Reindex ---------------- */
app.get("/health", (_req, res) => res.json({ ok: true }));
app.get("/kb-status", (_req, res) => {
  const embedded = KB.filter(x => Array.isArray(x.vec)).length;
  res.json({ chunks: KB.length, embedded, vocab: VOCAB_IDF.size });
});
app.post("/reindex", async (_req, res) => {
  res.json({ ok: true, msg: "Re-crawling started" });
  crawl().catch(e => console.log("reindex error", e?.message || String(e)));
});

/* ---------------- UI (no backticks; safe join) ---------------- */
app.get("/", (_q, res) => {
  const html = [
    '<!doctype html><html><head><meta charset="utf-8"/>',
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>',
    '<title>SODERBOT</title>',
    '<style>',
    ':root { --brand:#0ea5e9; --bg:#0F1115; --fg:#E8EAF0; --muted:#9AA1AC;',
    ' --inset-t: env(safe-area-inset-top, 0px);',
    ' --inset-b: env(safe-area-inset-bottom, 0px);',
    ' --inset-l: env(safe-area-inset-left, 0px);',
    ' --inset-r: env(safe-area-inset-right, 0px);',
    ' --gap: 16px; --vh: 1vh; }',
    '*{box-sizing:border-box} html,body{height:100%}',
    'body{margin:0;background:#0b0c10;color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}',
    '#chat-launcher{position:fixed;bottom:calc(var(--gap)+var(--inset-b));right:calc(var(--gap)+var(--inset-r));width:60px;height:60px;border-radius:50%;background:var(--brand);color:#fff;display:grid;place-items:center;font-size:24px;cursor:pointer;box-shadow:0 12px 30px rgba(0,0,0,.4);transition:transform .2s ease;z-index:2147483647}',
    '#chat-launcher:hover{transform:scale(1.05)}',
    '#panel{position:fixed;bottom:calc(90px + var(--inset-b));right:calc(20px + var(--inset-r));width:380px;max-height:72vh;background:#111319;color:#fff;border:1px solid #222634;border-radius:16px;display:none;flex-direction:column;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,.45);z-index:2147483646}',
    'header{padding:12px 16px;background:#0e1118;border-bottom:1px solid #1f2430;display:flex;align-items:center;justify-content:space-between}',
    'header .left{display:flex;align-items:center;gap:10px}',
    'header .dot{width:10px;height:10px;border-radius:50%;background:var(--brand)}',
    'header h1{margin:0;font-size:15px;font-weight:700;color:#fff;letter-spacing:.3px}',
    'select{background:#0f1320;color:#fff;border:1px solid #2a3344;border-radius:10px;padding:6px 8px;font-size:13px}',
    '#log{padding:14px;display:flex;flex-direction:column;gap:10px;overflow:auto}',
    '.msg{max-width:82%;padding:9px 12px;border-radius:12px;line-height:1.45;word-wrap:break-word}',
    '.user{margin-left:auto;background:#1b2330}',
    '.bot{margin-right:auto;background:#0f1723;border:1px solid #1f2937}',
    '.muted{color:var(--muted);font-size:12px;padding:0 2px}',
    'form{display:flex;gap:8px;padding:12px;border-top:1px solid #1f2430}',
    'input[type="text"]{flex:1;background:#0f1320;color:#fff;border:1px solid #22283a;border-radius:10px;padding:12px 14px;outline:none;min-height:44px}',
    'button{background:var(--brand);color:#fff;border:none;border-radius:10px;padding:0 16px;font-size:20px;cursor:pointer;min-height:44px;min-width:48px}',
    '@media (max-width:540px){',
    ' #panel{right:0;left:0;bottom:0;width:100vw;max-height:min(92dvh,92svh);max-height:calc(var(--vh)*100 - 8px);border-radius:16px 16px 0 0}',
    ' #chat-launcher{width:56px;height:56px;font-size:22px;bottom:calc(14px + var(--inset-b));right:calc(14px + var(--inset-r))}',
    ' header{padding:12px 14px} #log{padding:12px} .msg{max-width:88%} form{padding:10px}',
    '}',
    '@supports(padding:max(0px)){ #panel{ padding-bottom:max(0px, var(--inset-b)); } }',
    '</style>',
    '</head><body>',
    '<div id="chat-launcher" title="SODERBOT">💬</div>',
    '<div id="panel" role="dialog" aria-label="SODERBOT chat">',
    '  <header><div class="left"><div class="dot"></div><h1>SODERBOT</h1></div>',
    '    <select id="lang" aria-label="Language"><option value="en">English</option><option value="fi">Suomi</option><option value="sv">Svenska</option></select>',
    '  </header>',
    '  <div id="log" aria-live="polite"></div>',
    '  <form id="f" autocomplete="off"><input id="q" type="text" placeholder="Ask something…" inputmode="text"/><button type="submit" title="Send">➤</button></form>',
    '</div>',
    '<script>',
    'function setVH(){var vh=window.innerHeight*0.01;document.documentElement.style.setProperty("--vh",vh+"px");}',
    'setVH(); window.addEventListener("resize",setVH); window.addEventListener("orientationchange",setVH);',
    'var $panel=document.getElementById("panel"),$launch=document.getElementById("chat-launcher"),$log=document.getElementById("log"),$form=document.getElementById("f"),$q=document.getElementById("q"),$lang=document.getElementById("lang");',
    'function add(role,text){var d=document.createElement("div");d.className="msg "+(role==="user"?"user":"bot");d.innerHTML=(text||"").replace(/\\n/g,"<br>");$log.appendChild(d);$log.scrollTop=$log.scrollHeight;}',
    'function addMuted(t){var p=document.createElement("div");p.className="muted";p.textContent=t;$log.appendChild(p);$log.scrollTop=$log.scrollHeight;}',
    '$launch.onclick=function(){var open=$panel.style.display==="flex";$panel.style.display=open?"none":"flex";if(!open){setTimeout(function(){$q.focus();$log.scrollTop=$log.scrollHeight;},50);}};',
    '["resize","orientationchange"].forEach(function(evt){window.addEventListener(evt,function(){$log.scrollTop=$log.scrollHeight;},{passive:true});});',
    '$q.addEventListener("focus",function(){setTimeout(function(){$log.scrollTop=$log.scrollHeight;},150);});',
    '$form.addEventListener("submit",async function(e){',
    '  e.preventDefault(); var text=$q.value.trim(); if(!text)return;',
    '  $q.value=""; add("user",text); addMuted("Thinking…");',
    '  try{',
    '    var r=await fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:text,lang:$lang.value})});',
    '    var data=await r.json();',
    '    var m=document.querySelector(".muted:last-child"); if(m) m.remove();',
    '    add("assistant",data.reply||"(no answer)"); $log.scrollTop=$log.scrollHeight;',
    '  }catch(err){',
    '    var m2=document.querySelector(".muted:last-child"); if(m2) m2.remove();',
    '    add("assistant","Sorry — server error.");',
    '  }',
    '});',
    '</script>',
    '</body></html>'
  ].join("");
  res.setHeader("Content-Type", "text/html; charset=utf-8");
  res.send(html);
});

/* ---------------- Start ---------------- */
(async function main() {
  try {
    await crawl();
    console.log("✅ Knowledge built");
  } catch (e) {
    console.log("⚠️ Crawl error:", e?.message || String(e));
  }
  app.listen(PORT, () => console.log("✅ SODERBOT running on http://localhost:" + PORT));
})();
