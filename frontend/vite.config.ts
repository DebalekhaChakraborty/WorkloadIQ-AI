import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '');
  const backend = env.VITE_BACKEND_PROXY || 'http://127.0.0.1:8000';
  const port = Number(env.VITE_DEV_PORT || 3000);

  return {
    plugins: [react()],
    server: {
      host: '0.0.0.0',
      port,
      strictPort: true,
      proxy: {
        '/api': backend,
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: true,
    },
  };
});
