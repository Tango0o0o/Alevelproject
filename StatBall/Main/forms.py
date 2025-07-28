from django import forms # using the built in django forms class

class SignUpForm(forms.Form):
    email = forms.CharField(label="Email", max_length=100)
    password = forms.CharField(label="Password", widget=forms.PasswordInput())

   

    # This validates the email and field before allowing submission
    def is_valid_email(self):

        valid_tld = self.is_TLD()
        if valid_tld != True:
            return valid_tld
        
        # Checks if there is one "@" symbol
        if self.data.get("email").count("@") != 1:
            return "Email must contain one @"
        
        valid_domain = self.check_domain()
        if valid_domain != True:
            return valid_domain
        
        valid_local = self.check_local()
        if valid_local != True:
            return valid_local
        
        if len(self.data.get("email")) >= 100:
            return "Email must be less than 100 characters"
        
        return True
    
    # Checks if a password is valid based on criteria
    def is_valid_password(self):
        password = self.data.get("password").strip()
        
        if len(password) < 8:
            return "Password must be at least 8 characters"

        #Check for uppercase
        uppercase = False
        for letter in password:
            if letter.isupper():
                uppercase = True
                break
        
        if uppercase == False:
            return "Password must contain at least one uppercase letter"
        
        # Check for number 
        digit = False
        for letter in password:
            if letter.isdigit():
                digit = True
                break
        
        if digit == False:
            return "Password must contain at least one number"
        
        # Checks if each char is not a letter or number; hence special char
        alphanum = True 
        for letter in password:
            if not letter.isalnum():
                print("Alnum")
                alphanum = False
                break
        
        if alphanum == True:
            return "Password must contain at least one special character"
        
        return True

    # Checks if there is a valid TLD
    def is_TLD(self):
   
        if self.data.get("email").find(".com") == -1 and self.data.get("email").find(".uk") == -1:
            return "Email must be UK valid ending in .uk or .com"

        return True
    
    # Checks for a domain after the @ sign
    def check_domain(self):
        at_index = self.data.get("email").find("@")
        tld_index = self.data.get("email").find(".com")
        
        # Will be either .com or .uk, if finding .com return -1, then .uk must be in there
        if tld_index == -1:
            tld_index = self.data.get("email").find(".uk")
        
        # If the TLD and @ are next to each other, meaning nothing is inbetween, there is no domain
        if tld_index - at_index == 1: 
            return "Email must contain a domain after @"
        
        return True
    
    # Checks if there is a local domain before the "@" sign
    def check_local(self):
        if self.data.get("email").find("@") == 0:
            return "Email must contain a local domain before the '@' sign"
        return True
