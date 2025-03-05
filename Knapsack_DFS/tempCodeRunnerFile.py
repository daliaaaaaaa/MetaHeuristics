    start_time = time.time()
    max_value, selected_items = knapsack.solve()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
