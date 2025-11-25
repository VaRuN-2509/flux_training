import modal

# Replace "deployment-name" with your actual app name
f = modal.Function.from_name("train_kk", "run_training")

# Invoke it (detached style)
call_handle = f.spawn()  # starts the function without waiting
print(call_handle)   