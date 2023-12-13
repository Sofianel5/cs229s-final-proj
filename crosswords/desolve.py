from xdfile import parse
import os

path = 'data/xd'

for root, dirs, files in os.walk(path):
    for file in files:
        # check the extension of files
        if file.endswith('.xd'):
            filename = os.path.join(root, file)
            crossword = parse(filename)
            out_string = crossword.to_unicode(show_solved=False)
            out_file = filename.replace('data/xd/', 'data/xd-desolved/')
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, 'w') as f:
                f.write(out_string)
