import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  output: 'standalone',
  serverExternalPackages: ['@huggingface/transformers', 'onnxruntime-node', 'sharp'],
  outputFileTracingIncludes: {
    // Native bindings and the CJS build of onnxruntime-common are not reachable
    // through static analysis, so both runtime packages ship whole.
    '/**': [
      './node_modules/onnxruntime-node/**/*',
      './node_modules/onnxruntime-common/**/*',
    ],
  },
  poweredByHeader: false,
};

export default nextConfig;