import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

/** Static export: `npm run build` writes a plain website to out/ that opens anywhere, no server needed.
 *  outputFileTracingRoot pins the project root to this folder (a package-lock.json higher up made Next.js guess wrong). */
const nextConfig = {
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
  outputFileTracingRoot: dirname(fileURLToPath(import.meta.url)),
};
export default nextConfig;
