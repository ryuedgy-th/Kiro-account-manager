// Shared esbuild plugin for the headless service build + the proxyServer tests.
// Both compile src/main/proxy/proxyServer.ts with esbuild (electron-vite is NOT
// in that path), so both need this shim to teach esbuild Vite's `?raw`
// text-import convention (used for the portal/admin HTML templates).
import fs from 'fs'
import path from 'path'

/** Resolve `foo.html?raw` → file contents as a JS string default-export. */
export const rawPlugin = {
  name: 'raw-text-import',
  setup(build) {
    build.onResolve({ filter: /\?raw$/ }, args => {
      const clean = args.path.replace(/\?raw$/, '')
      return { path: path.resolve(args.resolveDir, clean), namespace: 'raw' }
    })
    build.onLoad({ filter: /.*/, namespace: 'raw' }, args => {
      const text = fs.readFileSync(args.path, 'utf8')
      return { contents: `export default ${JSON.stringify(text)};`, loader: 'js' }
    })
  }
}
