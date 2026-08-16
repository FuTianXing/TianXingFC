const http = require('http');
const fs = require('fs');
const path = require('path');
const root = 'D:/software/TianXingFC';
const server = http.createServer((req, res) => {
  const name = req.url === '/' ? 'visual_companion_5_figures.html' : decodeURIComponent(req.url.slice(1));
  const file = path.resolve(root, name);
  if (!file.startsWith(path.resolve(root)) || !fs.existsSync(file)) {
    res.writeHead(404); res.end('Not found'); return;
  }
  const data = fs.readFileSync(file);
  res.writeHead(200, {'Content-Type': file.endsWith('.html') ? 'text/html; charset=utf-8' : 'application/octet-stream', 'Content-Length': data.length});
  res.end(data);
});
server.listen(8766, '127.0.0.1', () => console.log('preview server listening on http://127.0.0.1:8766'));
