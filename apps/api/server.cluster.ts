import { spawn } from "bun";
import os from "os";

const numCPUs = os.cpus().length;

for (let i = 0; i < numCPUs; i++) {
  spawn(["bun", "run", "index.ts"], {
    env: { ...process.env, PORT: (5000 + i).toString() },
    stdio: ["inherit", "inherit", "inherit"],
  });
}

console.log(`Started ${numCPUs} Bun workers on ports 5000-${5000 + numCPUs - 1}`);