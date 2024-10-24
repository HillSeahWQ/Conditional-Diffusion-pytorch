# Determining the kernal size, stride, padding, ...etc for convolutional layers:

## Strided vs Fractional Strided Convolutions in CNNs

### Formula for Strided Convolution (Downsampling):
To calculate the output size of a **strided convolution** (used for **downsampling**), use the following formula:

$$
H_{out} = \frac{H_{in} - \text{kernel\_size} + 2 \times \text{padding}}{\text{stride}} + 1
$$

Where:
- $H_{out}$ = output height (or width),
- $H_{in}$ = input height (or width),
- **kernel\_size** = size of the convolution filter,
- **stride** = the step size of the filter,
- **padding** = amount of zero-padding added around the input.

#### Steps to Use:
1. **Start with Input Size**: Know the input feature map size, $H_{in}$.
2. **Choose Kernel Size**: Typically $3 \times 3$ or $5 \times 5$.
3. **Choose Stride**: Typically stride of 2 for downsampling.
4. **Determine Padding**: Padding = 1 or 2, depending on the kernel size.
5. **Use the Formula**: Plug in values and check $H_{out}$.

#### Example:
For an input of size $64 \times 64$, kernel size = 3, stride = 2, padding = 1:
$$
H_{out} = \frac{64 - 3 + 2 \times 1}{2} + 1 = \frac{63}{2} + 1 = 32
$$

---

### Formula for Fractional Strided Convolution (Upsampling):
For **fractional-strided convolutions** (used for **upsampling**, also called transposed convolutions), use this formula:

$$
H_{out} = (H_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel\_size} + \text{output\_padding}
$$

Where:
- $H_{out}$ = output height (or width),
- $H_{in}$ = input height (or width),
- **stride** = controls upsampling factor (e.g., stride of 2 doubles the input size),
- **kernel\_size** = size of the convolution filter,
- **padding** = amount of zero-padding added,
- **output\_padding** = extra padding added to the output.

#### Steps to Use:
1. **Start with Input Size**: Know the input size, $H_{in}$.
2. **Choose Stride**: Typically stride of 2 for doubling the size.
3. **Choose Kernel Size**: Often $3 \times 3$ or $5 \times 5$.
4. **Determine Padding**: Adjust based on desired output size.
5. **Check Output Padding**: Adjust to ensure the desired size.

#### Example:
For an input of size $16 \times 16$, kernel size = 5, stride = 2, padding = 2, and output padding = 1:
$$
H_{out} = (16 - 1) \times 2 - 2 \times 2 + 5 + 1 = 32
$$

---

### Key Differences:
1. **Strided Convolution**:
   - Used for **downsampling**.
   - **Reduces** the spatial dimensions.
   - Formula: 
   $$
   H_{out} = \frac{H_{in} - \text{kernel\_size} + 2 \times \text{padding}}{\text{stride}} + 1
  $$

2. **Fractional Strided (Transposed) Convolution**:
   - Used for **upsampling**.
   - **Increases** the spatial dimensions.
   - Formula: 
   $$
   H_{out} = (H_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel\_size} + \text{output\_padding}
   $$
