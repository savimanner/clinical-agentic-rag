INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Local Agentic RAG</title>
    <style>
      :root {
        color-scheme: light;
        --bg: #f7f3ea;
        --panel: #fffaf0;
        --ink: #1e1f18;
        --accent: #965d2f;
        --border: #d8c9ac;
      }
      body {
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", serif;
        background: radial-gradient(circle at top, #fff7df, var(--bg));
        color: var(--ink);
      }
      main {
        max-width: 980px;
        margin: 0 auto;
        padding: 32px 20px 48px;
      }
      h1 {
        margin: 0 0 8px;
        font-size: 2.4rem;
      }
      .grid {
        display: grid;
        gap: 16px;
        grid-template-columns: 1.7fr 1fr;
      }
      .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 8px 30px rgba(40, 31, 16, 0.07);
      }
      textarea, button {
        font: inherit;
      }
      textarea {
        width: 100%;
        min-height: 120px;
        box-sizing: border-box;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px;
        background: #fffdf7;
      }
      button {
        border: none;
        border-radius: 999px;
        padding: 10px 18px;
        background: var(--accent);
        color: white;
        cursor: pointer;
      }
      ul {
        padding-left: 18px;
      }
      pre {
        white-space: pre-wrap;
        background: #f3ead9;
        padding: 12px;
        border-radius: 12px;
        max-height: 360px;
        overflow: auto;
      }
      @media (max-width: 860px) {
        .grid { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <main>
      <h1>Local Agentic RAG</h1>
      <p>Ask questions against the curated corpus. The agent can search, retrieve, grade evidence, and iterate before answering.</p>
      <div class="grid">
        <section class="panel">
          <textarea id="question" placeholder="Ask a question about the indexed guidelines..."></textarea>
          <p><label><input type="checkbox" id="debug" /> Include debug trace</label></p>
          <button id="submit">Ask</button>
          <h2>Answer</h2>
          <pre id="answer">No answer yet.</pre>
          <h3>Citations</h3>
          <ul id="citations"></ul>
        </section>
        <aside class="panel">
          <h2>Library</h2>
          <ul id="library"></ul>
          <h2>Debug Trace</h2>
          <pre id="trace">Debug trace hidden.</pre>
        </aside>
      </div>
    </main>
    <script>
      async function loadLibrary() {
        const response = await fetch("/api/library");
        const docs = await response.json();
        const list = document.getElementById("library");
        list.innerHTML = "";
        for (const doc of docs) {
          const item = document.createElement("li");
          item.textContent = `${doc.doc_id} — ${doc.chunk_count} chunks${doc.indexed ? "" : " (not indexed)"}`;
          list.appendChild(item);
        }
      }

      async function askQuestion() {
        const question = document.getElementById("question").value;
        const debug = document.getElementById("debug").checked;
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({question, debug}),
        });
        const payload = await response.json();
        document.getElementById("answer").textContent = payload.answer || JSON.stringify(payload, null, 2);

        const citations = document.getElementById("citations");
        citations.innerHTML = "";
        for (const citation of payload.citations || []) {
          const item = document.createElement("li");
          item.textContent = `${citation.doc_id} :: ${citation.breadcrumbs} [${citation.chunk_id}]`;
          citations.appendChild(item);
        }

        document.getElementById("trace").textContent = payload.debug_trace
          ? JSON.stringify(payload.debug_trace, null, 2)
          : "Debug trace hidden.";
      }

      document.getElementById("submit").addEventListener("click", askQuestion);
      loadLibrary();
    </script>
  </body>
</html>
"""
