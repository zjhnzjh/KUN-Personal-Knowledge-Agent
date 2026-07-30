import { existsSync, readFileSync, unlinkSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const pidFile = resolve(projectRoot, ".run", "kun.pid");
if (!existsSync(pidFile)) {
  console.log("KUN is not running, or the PID file is missing.");
  process.exit(0);
}
const pid = Number(readFileSync(pidFile, "ascii").trim());
if (!Number.isInteger(pid) || pid <= 0) {
  console.error("Invalid KUN PID file.");
  process.exit(1);
}
const stopped = spawnSync("taskkill.exe", ["/PID", String(pid), "/T", "/F"], { windowsHide: true, encoding: "utf8" });
unlinkSync(pidFile);
if (stopped.status !== 0 && !String(stopped.stdout).includes("not found")) {
  console.error(stopped.stderr || stopped.stdout || "Unable to stop KUN.");
  process.exit(1);
}
console.log("KUN has been stopped.");
