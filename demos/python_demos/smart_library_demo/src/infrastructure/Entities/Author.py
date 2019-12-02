class Author:
    def __init__(self, author_id, first_name, last_name, middle_name):
        self.author_id = author_id
        self.first_name = first_name
        self.last_name = last_name
        self.middle_name = middle_name
        
    def _print(self):
        print(self.author_id)
        print(self.first_name)
        print(self.last_name)
        print(self.middle_name)