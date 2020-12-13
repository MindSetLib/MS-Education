import glob
files = glob.glob('\\*\\*.txt' )

with open( 'C:\\Users\\1\\Documents\\LibRu edu\\result.txt', 'w' ) as result:
    for file_ in files:
        for line in open( file_, 'r' ):
            result.write( line )