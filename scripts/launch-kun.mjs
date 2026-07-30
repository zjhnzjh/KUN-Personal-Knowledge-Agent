import { closeSync, existsSync, mkdirSync, openSync, readFileSync, writeFileSync } from "node:fs";
import { request } from "node:http";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const runDir = resolve(projectRoot, ".run");
const pidFile = resolve(runDir, "kun.pid");
const stdoutLog = resolve(runDir, "kun.out.log");
const stderrLog = resolve(runDir, "kun.err.log");
const appUrl = "http://127.0.0.1:3000/";

function probe(url, timeoutMs = 1200) {
  return new Promise((resolveProbe) => {
    const call = request(url, { method: "GET", timeout: timeoutMs }, (response) => {
      response.resume();
      resolveProbe((response.statusCode || 500) < 500);
    });
    call.once("timeout", () => { call.destroy(); resolveProbe(false); });
    call.once("error", () => resolveProbe(false));
    call.end();
  });
}

async function waitFor(urls, timeoutMs = 60000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if ((await Promise.all(urls.map((url) => probe(url)))).every(Boolean)) return;
    await new Promise((resolveWait) => setTimeout(resolveWait, 500));
  }
  throw new Error("KUN services did not become ready within 60 seconds");
}

function openBrowser(url) {
  const escaped = url.replace(/'/g, "''");
  const child = spawn("powershell.exe", ["-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", `Start-Process '${escaped}'`], {
    cwd: projectRoot, stdio: "ignore", windowsHide: true, detached: true,
  });
  child.unref();
}

mkdirSync(runDir, { recursive: true });
if (await probe("http://127.0.0.1:8765/api/health") && await probe(appUrl)) {
  openBrowser(appUrl);
  console.log("KUN is already running. The browser has been opened.");
  process.exit(0);
}

const out = openSync(stdoutLog, "a");
const error = openSync(stderrLog, "a");
const service = spawn(process.execPath, [resolve(projectRoot, "scripts", "start-dev.mjs")], {
  cwd: projectRoot,
  detached: true,
  windowsHide: true,
  stdio: ["ignore", out, error],
  env: { ...process.env, PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8" },
});
service.unref();
closeSync(out);
closeSync(error);
writeFileSync(pidFile, String(service.pid), "ascii");

try {
  await waitFor(["http://127.0.0.1:8765/api/health", appUrl]);
  console.log("KUN is ready. The browser will open automatically.");
} catch (reason) {
  console.error(reason instanceof Error ? reason.message : String(reason));
  console.error(`See log: ${stderrLog}`);
  process.exit(1);
}
