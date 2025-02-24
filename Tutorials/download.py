# import requests

# # URL of the file
# url = "https://home.strw.leidenuniv.nl/~daalen/Handin_files/Vandermonde.txt"

# # Send a GET request to download the file
# response = requests.get(url)

# # Check if the request was successful (status code 200)
# if response.status_code == 200:
#     # Save the file locally
#     with open("Vandermonde.txt", "wb") as file:
#         file.write(response.content)
#     print("Download complete: Vandermonde.txt")
# else:
#     print(f"Failed to download file. Status code: {response.status_code}")