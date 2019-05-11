
import os


def main():
	total_count = 0
	
	for subdir, dirnames, filenames, in os.walk('training_images'):
		loc_count = 0
		
		for f in filenames:
			f_name, f_ext = os.path.splitext(f)
			if f_ext == '.png':
				loc_count += 1
		
		readme = os.path.join(subdir, 'readme.txt')
		with open(readme, 'w') as f:
			f.write(f'Number of images: {loc_count}')
		
		total_count += loc_count
		
	with open('training_images/readme.txt', 'w') as f:
			f.write(f'Number of images: {total_count}')
		

if __name__ == '__main__': main()
