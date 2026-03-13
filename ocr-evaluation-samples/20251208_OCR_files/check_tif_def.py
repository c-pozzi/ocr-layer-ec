#!/usr/bin/env python3
"""
Inspect TIF and PDF files in a directory and compare their resolution.

Usage:
    python tif_inspector.py /path/to/directory
"""

import sys
from pathlib import Path
from PIL import Image

# Try to import PyMuPDF for PDF inspection
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


def get_tif_info(filepath: Path) -> dict:
    """Extract resolution info from a TIF file."""
    try:
        with Image.open(filepath) as img:
            raw_dpi = img.info.get("dpi", (None, None))
            # Convert DPI to float (handles IFDRational type)
            dpi = (float(raw_dpi[0]) if raw_dpi[0] else None,
                   float(raw_dpi[1]) if raw_dpi[1] else None)
            
            info = {
                "filename": filepath.name,
                "width": img.size[0],
                "height": img.size[1],
                "mode": img.mode,
                "dpi": dpi,
            }
            
            # Calculate physical size if DPI available
            if info["dpi"][0]:
                info["width_inches"] = info["width"] / info["dpi"][0]
                info["height_inches"] = info["height"] / info["dpi"][1]
            else:
                info["width_inches"] = None
                info["height_inches"] = None
                
            return info
    except Exception as e:
        return {"filename": filepath.name, "error": str(e)}


def get_pdf_info(filepath: Path) -> dict:
    """Extract embedded image info from a PDF file."""
    if not HAS_PYMUPDF:
        return {"filename": filepath.name, "error": "PyMuPDF not installed"}
    
    try:
        doc = fitz.open(str(filepath))
        
        images = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                
                images.append({
                    "page": page_num + 1,
                    "width": base_image["width"],
                    "height": base_image["height"],
                    "colorspace": base_image["colorspace"],
                    "bpc": base_image.get("bpc", "?"),  # bits per component
                    "size_kb": len(base_image["image"]) / 1024,
                })
        
        num_pages = len(doc)
        doc.close()
        
        return {
            "filename": filepath.name,
            "num_pages": num_pages,
            "images": images,
        }
    except Exception as e:
        return {"filename": filepath.name, "error": str(e)}


def main(directory: str):
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Error: Directory '{directory}' does not exist")
        sys.exit(1)
    
    # Find TIF files
    tif_files = list(dir_path.glob("*.tif")) + list(dir_path.glob("*.tiff"))
    tif_files = sorted(tif_files)
    
    # Find PDF files
    pdf_files = list(dir_path.glob("*.pdf"))
    pdf_files = sorted(pdf_files)
    
    # Build lookup by base name
    tif_by_name = {f.stem: f for f in tif_files}
    pdf_by_name = {f.stem: f for f in pdf_files}
    
    all_names = sorted(set(tif_by_name.keys()) | set(pdf_by_name.keys()))
    
    if not all_names:
        print(f"No TIF or PDF files found in {directory}")
        sys.exit(0)
    
    # Print TIF section
    print(f"{'='*120}")
    print(f"TIF FILES ({len(tif_files)} found)")
    print(f"{'='*120}")
    
    if tif_files:
        print(f"{'Filename':<45} {'Pixels':<15} {'DPI':<12} {'Size (inches)':<15} {'Mode'}")
        print("-" * 120)
        
        tif_stats = {"dpis": [], "widths": [], "heights": []}
        
        for tif_path in tif_files:
            info = get_tif_info(tif_path)
            
            if "error" in info:
                print(f"{info['filename']:<45} ERROR: {info['error']}")
                continue
            
            pixels = f"{info['width']} × {info['height']}"
            
            if info["dpi"][0]:
                dpi = f"{info['dpi'][0]:.0f} × {info['dpi'][1]:.0f}"
                size_inches = f"{info['width_inches']:.1f}\" × {info['height_inches']:.1f}\""
                tif_stats["dpis"].append(info["dpi"][0])
            else:
                dpi = "N/A"
                size_inches = "N/A"
            
            tif_stats["widths"].append(info["width"])
            tif_stats["heights"].append(info["height"])
            
            print(f"{info['filename']:<45} {pixels:<15} {dpi:<12} {size_inches:<15} {info['mode']}")
        
        print(f"\nTIF Summary: {len(tif_files)} files, ", end="")
        if tif_stats["widths"]:
            print(f"avg {sum(tif_stats['widths'])/len(tif_stats['widths']):.0f} × {sum(tif_stats['heights'])/len(tif_stats['heights']):.0f} px, ", end="")
        if tif_stats["dpis"]:
            unique_dpis = sorted(set(tif_stats["dpis"]))
            print(f"DPI: {', '.join(str(int(d)) for d in unique_dpis)}")
        else:
            print()
    else:
        print("No TIF files found")
    
    # Print PDF section
    print(f"\n{'='*120}")
    print(f"PDF FILES ({len(pdf_files)} found)")
    print(f"{'='*120}")
    
    if pdf_files:
        if not HAS_PYMUPDF:
            print("WARNING: PyMuPDF not installed. Install with: pip install pymupdf")
            print("Cannot inspect embedded images in PDFs.")
        else:
            print(f"{'Filename':<45} {'Embedded Image':<18} {'Colorspace':<12} {'BPC':<6} {'Size (KB)'}")
            print("-" * 120)
            
            pdf_stats = {"widths": [], "heights": []}
            
            for pdf_path in pdf_files:
                info = get_pdf_info(pdf_path)
                
                if "error" in info:
                    print(f"{info['filename']:<45} ERROR: {info['error']}")
                    continue
                
                if not info["images"]:
                    print(f"{info['filename']:<45} No embedded images")
                    continue
                
                for img in info["images"]:
                    pixels = f"{img['width']} × {img['height']}"
                    pdf_stats["widths"].append(img["width"])
                    pdf_stats["heights"].append(img["height"])
                    print(f"{info['filename']:<45} {pixels:<18} {img['colorspace']:<12} {img['bpc']:<6} {img['size_kb']:.1f}")
            
            print(f"\nPDF Summary: {len(pdf_files)} files, ", end="")
            if pdf_stats["widths"]:
                print(f"avg embedded image {sum(pdf_stats['widths'])/len(pdf_stats['widths']):.0f} × {sum(pdf_stats['heights'])/len(pdf_stats['heights']):.0f} px")
            else:
                print()
    else:
        print("No PDF files found")
    
    # Comparison section
    matching = set(tif_by_name.keys()) & set(pdf_by_name.keys())
    if matching and HAS_PYMUPDF:
        print(f"\n{'='*120}")
        print(f"TIF vs PDF COMPARISON ({len(matching)} matching pairs)")
        print(f"{'='*120}")
        print(f"{'Base Name':<40} {'TIF Pixels':<18} {'PDF Embedded':<18} {'Ratio'}")
        print("-" * 120)
        
        for name in sorted(matching):
            tif_info = get_tif_info(tif_by_name[name])
            pdf_info = get_pdf_info(pdf_by_name[name])
            
            if "error" in tif_info or "error" in pdf_info:
                continue
            
            if not pdf_info.get("images"):
                continue
            
            tif_px = f"{tif_info['width']} × {tif_info['height']}"
            pdf_img = pdf_info["images"][0]  # First image
            pdf_px = f"{pdf_img['width']} × {pdf_img['height']}"
            
            # Calculate ratio
            ratio_w = pdf_img['width'] / tif_info['width'] if tif_info['width'] else 0
            ratio_h = pdf_img['height'] / tif_info['height'] if tif_info['height'] else 0
            ratio = f"{ratio_w:.2f}x × {ratio_h:.2f}x"
            
            print(f"{name:<40} {tif_px:<18} {pdf_px:<18} {ratio}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tif_inspector.py <directory>")
        sys.exit(1)
    
    main(sys.argv[1])