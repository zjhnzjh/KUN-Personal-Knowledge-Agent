import { existsSync } from "node:fs";
import { spawn } from "node:child_process";
import { request } from "node:http";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function run(command, args, environment = {}) {
  return spawn(command, args, {
    cwd: projectRoot,
    stdio: "inherit",
    windowsHide: true,
    shell: process.platform === "win32" && command.toLowerCase().endsWith(".cmd"),
    env: { ...process.env, ...environment },
  });
}

function waitForHttp(url, timeoutMs = 30000) {
  const started = Date.now();
  return new Promise((resolveReady, reject) => {
    const check = () => {
      const probe = request(url, { method: "GET", timeout: 1200 }, (response) => {
        response.resume();
        if ((response.statusCode || 500) < 500) return resolveReady();
        retry();
      });
      probe.once("timeout", () => probe.destroy());
      probe.once("error", retry);
      probe.end();
    };
    const retry = () => {
      if (Date.now() - started >= timeoutMs) return reject(new Error(`等待 ${url} 超时`));
      setTimeout(check, 450);
    };
    check();
  });
}

function openBrowser(url) {
  const escapedUrl = url.replace(/'/g, "''");
  const opener = spawn(
    "powershell.exe",
    ["-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", `Start-Process '${escapedUrl}'`],
    {
      cwd: projectRoot,
      stdio: "ignore",
      windowsHide: true,
      shell: false,
    },
  );
  opener.once("error", (error) => {
    console.error(`无法调用默认浏览器：${error.message}`);
  });
  opener.unref();
}
async function waitFor(child) {
  return new Promise((resolveExit, reject) => {
    child.once("error", reject);
    child.once("exit", (code) => code === 0 ? resolveExit() : reject(new Error(`命令退出，代码 ${code}`)));
  });
}

if (!existsSync(resolve(projectRoot, "node_modules"))) {
  console.log("首次运行：正在安装前端依赖...");
  await waitFor(run("npm.cmd", ["install", "--registry=https://registry.npmmirror.com"]));
}

const defaultDataDir = resolve(process.env.LOCALAPPDATA || projectRoot, "KUN-AI-Infra");
const dataDir = process.env.KUN_DATA_DIR || defaultDataDir;
console.log(`KUN 正在启动本地 Agent 服务：数据目录 ${dataDir}`);
const backend = run(
  "python",
  ["-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8765"],
  { KUN_DATA_DIR: dataDir },
);
const frontend = run("npm.cmd", ["run", "dev"]);

let stopping = false;
function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  if (!backend.killed) backend.kill();
  if (!frontend.killed) frontend.kill();
  setTimeout(() => process.exit(code), 150);
}

backend.once("exit", (code) => {
  if (!stopping) {
    console.error(`本地 Agent 服务已停止（${code ?? "unknown"}）。`);
    stop(code || 1);
  }
});
frontend.once("exit", (code) => stop(code || 0));
process.on("SIGINT", () => stop(0));
process.on("SIGTERM", () => stop(0));

console.log("正在等待前后端就绪，完成后会自动打开浏览器...");
Promise.all([
  waitForHttp("http://127.0.0.1:8765/api/health"),
  waitForHttp("http://127.0.0.1:3000/"),
]).then(() => {
  console.log("KUN 已就绪：http://127.0.0.1:3000");
  openBrowser("http://127.0.0.1:3000/");
}).catch((error) => {
  console.error(`自动打开失败：${error.message}`);
  console.error("请手动访问：http://127.0.0.1:3000/");
});
console.log("按 Ctrl+C 可同时停止前后端。");
