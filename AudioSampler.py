import librosa
import numpy as np

def scale_values(values, new_min=0, new_max=32):
    # Apply logarithmic scaling
    values = np.log1p(values - np.min(values))
    
    # Find the min and max values in the scaled values
    old_min = np.min(values)
    old_max = np.max(values)
    
    # Linearly scale and shift the values to the desired range [new_min, new_max]
    scaled_values = (values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    
    return scaled_values

def calculate_band_averages(y, sr, bands, n_fft=2048, hop_length=4800):
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    
    bin_ranges = [
        (int(np.floor(f1 * n_fft / sr)), int(np.floor(f2 * n_fft / sr)))
        for f1, f2 in bands
    ]
    
    levels = [[] for _ in range(len(bands))]
    
    for t in range(D.shape[1]):
        for i, (bin_start, bin_end) in enumerate(bin_ranges):
            level_data = D[bin_start:bin_end, t]
            if level_data.size > 0:
                levels[i].append(np.mean(level_data))
            else:
                levels[i].append(0)
    
    return np.array(levels).T

def array_to_lua_table(arr, table_name, decimal_places=2, values_per_line=60):
    lua_table = f"{table_name} = {{\n"
    for i, item in enumerate(arr.T):
        lua_table += f"\tLevel{i+1} = {{\n"
        for j, val in enumerate(item):
            if decimal_places == 0:
                lua_table += f"{int(round(val))}, "
            else:
                lua_table += f"{round(val, decimal_places)}, "
            
            if (j + 1) % values_per_line == 0:
                lua_table += "\n"
        lua_table = lua_table.rstrip(", ") + "\n\t},\n"
    lua_table += "}"
    return lua_table

# Example usage
file_path = "Nuclear_Winter_combat.ogg"
y, sr = librosa.load(file_path, sr=None)

bands = [
    (16, 60),
    (60, 250),
    (250, 500),
    (500, 2000),
    (2000, 4000),
    (4000, 6000),
    (6000, 20000)
]

band_averages = calculate_band_averages(y, sr, bands)
scaled_band_averages = scale_values(band_averages)

# Convert to Lua table
table_name = "NuclearWinterCombat"
lua_table_str = array_to_lua_table(scaled_band_averages, table_name, 0)

# Output Lua table to a file
output_file_path = "audio_data.lua"
with open(output_file_path, "w") as file:
    file.write(lua_table_str)
