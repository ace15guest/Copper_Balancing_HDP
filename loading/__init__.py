import gerber
from gerber.render import RenderSettings, GerberCairoContext
import cairosvg

def gerber_to_png(gerber_path, png_path):
    # Read the Gerber file
    gerber_data = gerber.read(gerber_path)

    # Create a new drawing context
    ctx = GerberCairoContext()

    # Render the Gerber file to SVG
    svg_data = ctx.render_layer(gerber_data, 'output.svg', settings=RenderSettings())

    # Convert SVG to PNG
    cairosvg.svg2png(url='output.svg', write_to=png_path)

# Example usage
gerber_file = '/path/to/your/file.274x'
png_file = '/path/to/output/file.png'
gerber_to_png(gerber_file, png_file)