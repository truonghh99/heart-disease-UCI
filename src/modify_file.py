filePath = 'modified-data/switzerland.data'

# read file
with open(filePath, errors='ignore') as f:
  file = f.read()

# remove endline and mark 'name' as new splitting points
file = file.replace('\n', ' ')
file = file.replace('name ', 'name\n')

with open(filePath, 'w') as f:
	f.write(file)

# read new file line by line
with open(filePath, errors='ignore') as f:
  lines = f.readlines()

# write each line back to file, ignore lines that don't have exact 76 values
with open(filePath, 'w') as f:
	for line in lines:
		if line.count(' ') == 75:
			f.write(line)