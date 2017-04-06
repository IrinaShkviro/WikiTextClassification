# -*- coding: utf-8 -*-

import os, shlex, subprocess, sys

tool_dir = "fields_of_mathematics"
os.chdir(tool_dir)

finished = 0
dir_size = 34937

for p in os.listdir():
    command_line = ('pandoc -s -f mediawiki -t plain -o ../plains/%s %s'
                       % (p, p) )
    args = shlex.split(command_line)
    subprocess.Popen(args)
    finished = finished + 1
    sys.stdout.write("%i / %i \r" % (finished, dir_size))