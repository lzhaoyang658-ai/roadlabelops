import { cpSync, existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const standaloneRoot = join(".next", "standalone");
const standaloneNext = join(standaloneRoot, ".next");

mkdirSync(standaloneNext, { recursive: true });
cpSync(join(".next", "static"), join(standaloneNext, "static"), { recursive: true });

if (existsSync("public")) {
  cpSync("public", join(standaloneRoot, "public"), { recursive: true });
}

console.log("Prepared standalone runtime with static assets.");
