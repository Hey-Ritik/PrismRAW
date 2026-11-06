import {mkdir, copyFile, cp, writeFile} from 'node:fs/promises';

const value = process.env.PUBLIC_API_BASE_URL;
if (!value) throw new Error('Set PUBLIC_API_BASE_URL to the HTTPS Render backend origin.');
const url = new URL(value);
if (url.protocol !== 'https:' || url.username || url.password || url.pathname !== '/' || url.search || url.hash) {
  throw new Error('PUBLIC_API_BASE_URL must be an HTTPS origin without a path, credentials, query or fragment.');
}
await mkdir('dist', {recursive: true});
await copyFile('index.html', 'dist/index.html');
await cp('static', 'dist/static', {recursive: true});
await writeFile('dist/static/config.js', `window.PRISMRAW_API_BASE = ${JSON.stringify(url.origin)};\n`);
console.log(`Built static frontend; API: ${url.origin}`);
