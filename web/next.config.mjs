/** Static export: `npm run build` writes a plain website to out/ that opens anywhere, no server needed. */
const nextConfig = { output: "export", images: { unoptimized: true }, trailingSlash: true };
export default nextConfig;
