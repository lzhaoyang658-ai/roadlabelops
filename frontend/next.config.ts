import type { NextConfig, SizeLimit } from "next";

const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8100";
// Keep the Next proxy buffer aligned with ROADLABELOPS_MAX_UPLOAD_BYTES (2 GiB by default).
// Override with a Next-supported size such as "512mb" only when the backend limit matches.
const configuredProxyBodySize = process.env.NEXT_PROXY_CLIENT_MAX_BODY_SIZE ?? "2gb";
if (!/^\d+(?:\.\d+)?(?:b|kb|mb|gb)$/i.test(configuredProxyBodySize)) {
  throw new Error("NEXT_PROXY_CLIENT_MAX_BODY_SIZE must use b, kb, mb, or gb units");
}
const proxyClientMaxBodySize = configuredProxyBodySize.toLowerCase() as SizeLimit;

const nextConfig: NextConfig = {
  output: "standalone",
  experimental: {
    proxyClientMaxBodySize,
  },
  async rewrites() {
    return [{ source: "/api/v1/:path*", destination: `${backendUrl}/api/v1/:path*` }];
  },
};

export default nextConfig;
