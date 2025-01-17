from google.oauth2.service_account import Credentials

credentials = Credentials.from_service_account_file("kdsh-pathway-bab3103d5539.json",
                                                        scopes=["https://www.googleapis.com/auth/drive"])
print(credentials)