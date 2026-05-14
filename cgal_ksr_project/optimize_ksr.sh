#!/bin/bash

# ============================================================
# KSR Parameter Optimization Script
# Tests different combinations of lambda, minpts, and k
# ============================================================

# Input file
INPUT_PLY="/mnt/d/PFTMeng/cluster_208_export/cluster_208_lego_mesh.ply"

# Output directory
OUTPUT_DIR="/mnt/d/PFTMeng/cluster_208_export/Kinetic_Shapes"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# KSR executable path
KSR_BIN="$HOME/garfield/cgal_ksr_project/build/ksr_building"

# ============================================================
# Parameter ranges to test
# ============================================================

# Lambda values (complexity vs fidelity)
LAMBDAS=(0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Minpts values (minimum points per region)
MINPTS=(50 100 150 200 300 500)

# K values (partition complexity)
KS=(1 2)

# ============================================================
# Run tests
# ============================================================

echo "============================================================"
echo "KSR Parameter Optimization"
echo "============================================================"
echo "Input: $INPUT_PLY"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Results log file
LOG_FILE="$OUTPUT_DIR/optimization_results.txt"
echo "KSR Parameter Optimization Results" > "$LOG_FILE"
echo "Input: $INPUT_PLY" >> "$LOG_FILE"
echo "Date: $(date)" >> "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Counter for progress
total_tests=$((${#LAMBDAS[@]} * ${#MINPTS[@]} * ${#KS[@]}))
current_test=0

for lambda in "${LAMBDAS[@]}"; do
    for minpts in "${MINPTS[@]}"; do
        for k in "${KS[@]}"; do
            current_test=$((current_test + 1))
            
            # Output filename
            output_name="ksr_l${lambda}_m${minpts}_k${k}.ply"
            output_path="$OUTPUT_DIR/$output_name"
            
            echo "[$current_test/$total_tests] Testing: lambda=$lambda, minpts=$minpts, k=$k"
            
            # Run KSR and capture output
            result=$("$KSR_BIN" "$INPUT_PLY" \
                -lambda "$lambda" \
                -minpts "$minpts" \
                -k "$k" \
                -output "$output_path" 2>&1)
            
            # Extract key metrics from output
            planes=$(echo "$result" | grep "planar shapes" | grep -oP '\d+(?= planar)')
            vertices=$(echo "$result" | grep "Vertices:" | grep -oP '\d+')
            faces=$(echo "$result" | grep "Faces:" | grep -oP '\d+')
            
            # Check if successful
            if [ -f "$output_path" ]; then
                status="SUCCESS"
                file_size=$(du -h "$output_path" | cut -f1)
            else
                status="FAILED"
                file_size="N/A"
                planes="N/A"
                vertices="N/A"
                faces="N/A"
            fi
            
            # Log results
            echo "  -> Planes: $planes, Faces: $faces, Status: $status"
            echo "lambda=$lambda, minpts=$minpts, k=$k | Planes: $planes | Faces: $faces | Vertices: $vertices | Status: $status | File: $output_name" >> "$LOG_FILE"
            
        done
    done
done

echo ""
echo "============================================================"
echo "Optimization complete!"
echo "Results saved to: $LOG_FILE"
echo "Meshes saved to: $OUTPUT_DIR"
echo "============================================================"

# Print summary
echo ""
echo "=== Summary of Results ==="
cat "$LOG_FILE"
