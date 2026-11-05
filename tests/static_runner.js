// Runs API requests through web/engine.js in Node and prints the JSON results.
// Usage: node tests/static_runner.js <model_dir> <requests.json>
const fs = require("fs");
const path = require("path");
const { Engine } = require(path.join(__dirname, "..", "web", "engine.js"));

const [dir, reqFile] = process.argv.slice(2);
const meta = JSON.parse(fs.readFileSync(path.join(dir, "meta.json"), "utf8"));
const movies = JSON.parse(fs.readFileSync(path.join(dir, "movies.json"), "utf8"));
const bin = fs.readFileSync(path.join(dir, "model.bin"));
const buffer = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
const engine = new Engine(meta, movies, buffer);

const out = JSON.parse(fs.readFileSync(reqFile, "utf8")).map(({ path: p, body }) => {
  try {
    return engine.handle(p, body ? { method: "POST", body: JSON.stringify(body) } : {});
  } catch (e) {
    return { error: e.message, status: e.status };
  }
});
process.stdout.write(JSON.stringify(out));
