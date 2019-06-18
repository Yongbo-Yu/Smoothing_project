



class Parser:
    def __init__(self, path):
	    self.file = open(path, "r")
	    self.unparsed_info = self.file.read()
	    self.element_list = [] # Make an empty list

    def parse_file(self, string, delimiter):
    	for elements in string.split(delimiter):
      	  e = elements.strip(delimiter)
       	  if e != '': # Check for empty strings
           	 self.element_list.append(e) #Append to list 

    def print_unparsed(self):
        print(self.unparsed_info)

    def print_parsed(self):
        print(self.element_list)

    def close_file(self):
        self.file.close()

        