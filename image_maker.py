'''Script that chooses which image-making subscript to run.'''

import os
import sys


def main():
	plat = sys.platform
	
	if plat == 'ios':
		import _mob_image_maker as img_maker
	elif plat == 'darwin':
		import _comp_image_maker as img_maker
		
	img_maker.main()
		

if __name__ == '__main__': main()
