class User:
    def __init__(self, user_id, phone, first_name, last_name,
                 middle_name = None):
        self.user_id = user_id
        self.phone = phone
        self.first_name = first_name
        self.last_name = last_name
        self.middle_name = middle_name
        
    def _print(self):
        print(self.user_id)
        print(self.phone)
        print(self.first_name)
        print(self.last_name)
        print(self.middle_name)