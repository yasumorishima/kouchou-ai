import path from "node:path";
import type { NextConfig } from "next";
import { buildCspHeaderValue } from "../shared/csp";

// #885: local desktop / single-binary distribution build.
// When NEXT_PUBLIC_OUTPUT_MODE=export, emit static assets only (no Node runtime);
// the admin UI then talks to the FastAPI backend directly from the client.
// Mirrors apps/public-viewer/next.config.ts so both apps share one toggle.
const isStaticExport = process.env.NEXT_PUBLIC_OUTPUT_MODE === "export";
const BASE_PATH = process.env.NEXT_PUBLIC_STATIC_EXPORT_BASE_PATH || "";
const DIST_DIR = process.env.STATIC_EXPORT_DIST_DIR || ".next";

const enableGoogleAnalytics = Boolean(process.env.NEXT_PUBLIC_ADMIN_GA_MEASUREMENT_ID);
const contentSecurityPolicy = buildCspHeaderValue({
  apiBasePath: process.env.API_BASEPATH,
  publicApiBasePath: process.env.NEXT_PUBLIC_API_BASEPATH,
  siteUrl: process.env.NEXT_PUBLIC_SITE_URL,
  enableGoogleAnalytics,
  isDevelopment: process.env.NODE_ENV !== "production",
});

const nextConfig: NextConfig = {
  output: isStaticExport ? "export" : "standalone",
  basePath: isStaticExport ? BASE_PATH : "",
  assetPrefix: isStaticExport ? BASE_PATH : "",
  distDir: isStaticExport ? DIST_DIR : ".next",
  outputFileTracingRoot: path.join(__dirname, "../../"),
  experimental: {
    optimizePackageImports: ["@chakra-ui/react"],
    serverActions: {
      bodySizeLimit: "100mb",
    },
  },
  serverExternalPackages: ["fs", "path"],
  // CSP is sent as a response header only in the server (standalone) runtime.
  // In static export there is no Node server, so CSP must be applied by the
  // static host (FastAPI / desktop shell). headers() is unsupported with
  // output: "export", so it is omitted in that mode.
  ...(isStaticExport
    ? {}
    : {
        async headers() {
          return [
            {
              source: "/:path*",
              headers: [
                {
                  key: "Content-Security-Policy",
                  value: contentSecurityPolicy,
                },
              ],
            },
          ];
        },
      }),
};

export default nextConfig;
