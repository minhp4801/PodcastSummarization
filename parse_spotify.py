import json
import os

def parse(filepath):
    in_fn = filepath
    out_fn = os.path.splitext(filepath)[0] + '.txt'
    f_in = open(in_fn, 'r')
    f_out = open(out_fn, 'w') 
    parsed = json.loads(f_in.read())
    parsed = parsed['results']
    for segment in parsed:
        if 'transcript' in segment['alternatives'][0]:
            f_out.write(segment['alternatives'][0]['transcript'])
    
    f_in.close()
    f_out.close()
    
if __name__ == "__main__":
    source_dir = "./data/podcasts-transcripts-summarization-testset"
    files = os.listdir(source_dir)
    for f in files:
        parse(os.path.join(source_dir, f))