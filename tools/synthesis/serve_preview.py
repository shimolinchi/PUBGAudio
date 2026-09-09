"""Serve local audition files with HTTP byte ranges for reliable media seeking."""
import argparse
import csv
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        origin=f'http://127.0.0.1:{self.server.server_port}'
        if self.path!='/feedback/export' or self.headers.get('Origin')!=origin or self.headers.get('Content-Type','').split(';')[0]!='application/json':
            self.send_error(403);return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=262144: raise ValueError('Invalid length')
            rows=json.loads(self.rfile.read(length))
            known={json.loads(line)['id'] for line in (Path(self.directory)/'recipes.jsonl').read_text(encoding='utf-8').splitlines()}
            if not isinstance(rows,list) or {r['id'] for r in rows}!=known or len(rows)!=len(known): raise ValueError('Invalid sample list')
            if any(not isinstance(r.get(k,''),str) for r in rows for k in ['id','natural','direction','notes']): raise ValueError('Invalid feedback')
        except (ValueError,KeyError,TypeError):
            self.send_error(400,'Invalid feedback');return
        destination=Path(self.directory)/'feedback';destination.mkdir(exist_ok=True)
        name='feedback-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.csv'
        output=io.StringIO();writer=csv.writer(output)
        writer.writerow(['样本','自然程度','方向','备注'])
        for row in rows:
            values=[row.get(k,'') for k in ['id','natural','direction','notes']]
            writer.writerow(["'"+v if v.startswith(('=','+','-','@','\t')) else v for v in values])
        (destination/name).write_text(output.getvalue(),encoding='utf-8-sig',newline='')
        payload=json.dumps(dict(url='feedback/'+name,file='feedback/'+name)).encode('utf-8')
        self.send_response(201);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)

    def send_head(self):
        path=Path(self.translate_path(self.path))
        self.remaining=None
        if not path.is_file(): return super().send_head()
        stream=path.open('rb');size=path.stat().st_size
        raw=self.headers.get('Range');begin=0;end=size-1
        if raw:
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',raw.strip())
            if not match or not any(match.groups()):
                stream.close();self.send_error(416,'Invalid byte range');return None
            left,right=match.groups()
            if left:
                begin=int(left);end=min(int(right),size-1) if right else size-1
            else:
                begin=max(0,size-int(right))
            if begin>end or begin>=size:
                stream.close();self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return None
        self.send_response(206 if raw else 200)
        self.send_header('Content-Type','audio/wav' if path.suffix.lower()=='.wav' else self.guess_type(str(path)))
        self.send_header('Accept-Ranges','bytes')
        self.send_header('Content-Length',str(end-begin+1))
        self.send_header('Last-Modified',self.date_time_string(path.stat().st_mtime))
        if raw: self.send_header('Content-Range',f'bytes {begin}-{end}/{size}')
        self.end_headers();stream.seek(begin);self.remaining=end-begin+1
        return stream

    def copyfile(self,source,output):
        if self.remaining is None: return super().copyfile(source,output)
        try:
            while self.remaining>0:
                data=source.read(min(65536,self.remaining))
                if not data: break
                output.write(data);self.remaining-=len(data)
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):
            pass  # Browsers routinely cancel an old range when seeking or changing clips.


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('root',type=Path);a.add_argument('--port',type=int,default=8766)
    args=a.parse_args();root=args.root.resolve(strict=True)
    if not root.is_dir(): raise ValueError('Preview root must be a directory')
    server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(Handler,directory=str(root)))
    print(f'Local preview: http://127.0.0.1:{args.port}/',flush=True)
    server.serve_forever()


if __name__=='__main__':main()
