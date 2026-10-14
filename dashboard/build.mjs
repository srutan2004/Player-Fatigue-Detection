import { build } from "esbuild";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const root = dirname(fileURLToPath(import.meta.url));
await build({
  entryPoints: [resolve(root, "app.jsx")],
  outfile: resolve(root, "assets/app.js"),
  bundle: true,
  minify: true,
  jsx: "automatic",
  define: { "process.env.NODE_ENV": '"production"' },
  target: ["es2022"],
  legalComments: "eof",
});
console.log("Built offline React dashboard client.");
