class Book:
    def __init__(self, book_id, file_path, title, year, publisher, authors):
        self.book_id = book_id
        self.file_path = file_path
        self.title = title
        self.year = year
        self.publisher = publisher
        self.authors = authors
        
    def _print(self):
        print(self.book_id)
        print(self.file_path)
        print(self.title)
        print(self.year)
        print(self.publisher)
        print("AUTHORS:")
        i = 0
        for elem in self.authors:
            (self.authors[i])._print()
            i = i + 1