# def process_file(input_path, output_path):
#     seen = set()
#     unique_lines = []

#     with open(input_path, 'r') as file:
#         for line in file:
#             # Extract the part before any spaces (the "path" part)
#             path_part = line.split()[0]
            
#             # If this path has not been seen, add it to the set and list
#             if path_part not in seen:
#                 seen.add(path_part)
#                 unique_lines.append(path_part)
    
#     # lines = [line.replace('_color_all', '') for line in unique_lines]
#     # processed_lines = [line.replace('./frames/', '') for line in lines]

#     with open(output_path, 'w') as file:
#         for line in unique_lines:
#             file.write(f"{line}\n")

# # Example usage
# input_path = '/home/user/github/mypyskl/data/LD/LDall.list'
# output_path = '/home/user/github/mypyskl/data/LD/LDall_videos.list'
# process_file(input_path, output_path)

###ion
def process_file(input_path, output_path):
    seen = set()
    unique_lines = []

    with open(input_path, 'r') as file:
        for line in file:
            # Extract the part before any spaces (the "path" part)
            path_part = line.split()[0]
            
            # If this path has not been seen, add it to the set and list
            if path_part not in seen:
                seen.add(path_part)
                unique_lines.append(path_part)
    
    # lines = [line.replace('_color_all', '') for line in unique_lines]
    # processed_lines = [line.replace('./frames/', '') for line in lines]

    with open(output_path, 'w') as file:
        for line in unique_lines:
            file.write(f"{line}\n")

# Example usage
input_path = '/home/user/github/mypyskl/data/ipn/ipnall.list'
output_path = '/home/user/github/mypyskl/data/ipn/ipnall_videos.list'
process_file(input_path, output_path)
