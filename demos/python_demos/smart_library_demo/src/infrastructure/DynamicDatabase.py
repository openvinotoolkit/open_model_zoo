from datetime import datetime, date, time
from Entities.User import *
from Entities.Author import *
from Entities.Book import *

class BorrowedBooks():
    def __init__(self, bookID, userID, borrowed, bdate, rdate):
        self.bookID = bookID
        self.userID = userID
        self.borrowed = borrowed
        self.bdate = bdate
        self.rdate = rdate

class DynamicDB():
    
    def __init__(self):
        self.Users = []
        self.Books = []
        self.BBooks = []

    def add_user(self, userID):
        user = User(userID, '', 'User# ' +str(userID), '', '')
        user._print()
        self.Users.append(user)
    
    def delete_user(self):
        ''' '''
    
    def add_book(self, bookID, title, author,  publisher, date):
        lAuthors = []
        authors = author.split(', ')
        for a in authors:
            a = a.replace('.', '')
            names = a.split(' ')
            if len(names) == 3:
                lAuthors.append(Author(-1, names[0], names[2], names[1]))
            elif len(names) == 2:
                lAuthors.append(Author(-1, names[0], names[1], ''))
            else:
                Exception('Book`s authors not in correct format')

        book = Book(bookID, '', title, date, publisher, lAuthors)
        self.Books.append(book)
    
    def delete_book(self):
        ''' '''    
     
    def get_ret_book(self, userID, bookID):
        dateNow = str(datetime.now()).split(' ')[0]
        find = False
        isBorrowed = False

        for book in self.BBooks:
            if book.bookID ==  bookID:
                if book.borrowed:
                    find = True
                    isBorrowed = book.borrowed
                    if userID == book.userID:
                        book.borrowed = not book.borrowed
                        book.rdate = dateNow
                        print('Book returned succesfully')
                    else:
                        print('This book is not on your account')

        if not find and not isBorrowed:
            for book in self.Books: 
                if book.book_id ==  bookID:
                        bbook = BorrowedBooks(bookID, userID, True, dateNow, '' )
                        find = True
                        self.BBooks.append(bbook)
                        print("len =", len(self.BBooks))
                        print('Book borrowed succesfully')
        if not find:
           print('There is no such book')
            
        return find
           
    def print_users(self):
        print('{:<10}{:<10}'.format('ID', 'Name'))
        for user in self.Users:
            print('{:<10}{:<10}'.format(user.user_id, user.first_name))


    def print_books(self):
        authorsStr = ''
        print('{:<10}{:<30}{:<40}{:<10}{:<20}'.format('ID', 'Author','Title',
        'Publisher', 'Publication date'))
        for book in self.Books:
            authorsStr = ''
            for a in book.authors:
                if(a.middle_name != ''):
                    authorsStr += (a.first_name[0] + '. ' +
                                a.middle_name[0] + '. ' 
                                + a.last_name)
                else:
                    authorsStr += (a.first_name[0] + '. ' + 
                                   a.last_name)
                if (a != book.authors[-1]):
                    authorsStr += ', '

            print('{:<10}{:<30}{:<40}{:<10}{:<20}'.format(book.book_id, authorsStr,
                book.title, book.publisher, book.year))

    def print_borrowed_books(self):
        print('{:<10}{:<10}{:<20}{:<40}{:<20}{:<20}'.format('User ID', 'Book ID', 'First name',
             'Title',  'Borrow date','Return date'))
        for bbook in self.BBooks:
            for book in self.Books:
                for user in self.Users:
                    if bbook.userID == user.user_id and bbook.bookID == book.book_id:
                        print('{:<10}{:<10}{:<20}{:<40}{:<20}{:<20}'.format(user.user_id, book.book_id,
                                   user.first_name, book.title, bbook.bdate, bbook.rdate))
