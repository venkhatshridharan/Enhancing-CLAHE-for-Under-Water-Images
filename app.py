import gradio as gr 
from processes import * 


def processing_pipeline(input_image):
    # Convert Gradio image to OpenCV format
    input_image = input_image.astype('uint8')

    # Apply CLAHE
    clahe_image = clahe(input_image)

    # Apply sharpening
    sharpened_image = sharpen(clahe_image, sharpen_strength=0.5)

    # Apply denoising
    onlyDenoising_image = denoise(clahe_image)

    # Apply sharpening + denoising
    denoised_image = denoise(sharpened_image, strength=10)

    # apply denoiseing + shaprening
    denoiseSharp_image = sharpen(denoise(input_image))

    # Generate histograms
    original_hist = plot_histogram(input_image)
    clahe_hist = plot_histogram(clahe_image)
    sharpened_hist = plot_histogram(sharpened_image)
    denoised_hist = plot_histogram(denoised_image)
    denoiseSharp_hist = plot_histogram(denoiseSharp_image)
    onlyDenoising_hist = plot_histogram( onlyDenoising_image)

    # Calculate SNR, CNR, and PSNR for each image
    snr_original = "Cannot Calculate"
    snr_clahe = calculate_snr(clahe_image, input_image - clahe_image)
    snr_sharpened = calculate_snr(sharpened_image, input_image - sharpened_image)
    snr_denoised = calculate_snr(denoised_image, input_image - denoised_image)
    snr_denoiseSharp = calculate_snr(denoiseSharp_image,input_image - denoiseSharp_image)
    snr_onlyDenoise = calculate_snr(onlyDenoising_image,input_image - onlyDenoising_image)

    cnr_original = "Cannot Calculate"
    cnr_clahe = calculate_cnr(clahe_image, input_image - clahe_image)
    cnr_sharpened = calculate_cnr(sharpened_image, input_image - sharpened_image)
    cnr_denoised = calculate_cnr(denoised_image, input_image - denoised_image)
    cnr_denoiseSharp = calculate_cnr(denoiseSharp_image,input_image - denoiseSharp_image)
    cnr_onlyDenoise = calculate_cnr(onlyDenoising_image,input_image - onlyDenoising_image)

    psnr_original = "Cannot Calculate"
    psnr_clahe = calculate_psnr(clahe_image, input_image)
    psnr_sharpened = calculate_psnr(sharpened_image, input_image)
    psnr_denoised = calculate_psnr(denoised_image, input_image)
    psnr_denoiseSharp = calculate_psnr(denoiseSharp_image,input_image - denoiseSharp_image)
    psnr_onlyDenoise = calculate_psnr(onlyDenoising_image,input_image - onlyDenoising_image)



    return (input_image, clahe_image, sharpened_image, denoised_image,denoiseSharp_image,onlyDenoising_image,
            original_hist, clahe_hist, sharpened_hist, denoised_hist,denoiseSharp_hist,onlyDenoising_hist ,
            snr_original, snr_clahe, snr_sharpened, snr_denoised,snr_denoiseSharp,snr_onlyDenoise,
            cnr_original, cnr_clahe, cnr_sharpened, cnr_denoised, cnr_denoiseSharp, cnr_onlyDenoise,
            psnr_original, psnr_clahe, psnr_sharpened, psnr_denoised, psnr_denoiseSharp,psnr_onlyDenoise )


with gr.Blocks() as demo:

    inputImg = gr.Image(label="Input Image")
    processs = gr.Button("Process")
    gr.ClearButton()

    with gr.Column():
        # Original Image
        with gr.Row():
            org_image = gr.Image(label="Original")
            org_image_histogram = gr.Image(label="Original Histogram")
            with gr.Column():
                org_image_snr = gr.Textbox(label="SNR - Original")
                org_image_cnr = gr.Textbox(label="CNR - Original")
                org_image_psnr = gr.Textbox(label="PSNR - Original")

        with gr.Row():
            # CLAHE
            clahe_image = gr.Image(label="CLAHE")
            clahe_image_histogram = gr.Image(label="CLAHE Histogram")
            with gr.Column():
                clahe_image_snr = gr.Textbox(label="SNR - CLAHE")
                clahe_image_cnr = gr.Textbox(label="CNR - CLAHE")
                clahe_image_psnr = gr.Textbox(label="PSNR - CLAHE")

        with gr.Row():
            # CLAHE + Sharpen
            clahe_sharpen_image = gr.Image(label="CLAHE + Sharpen")
            clahe_sharpen_image_histogram = gr.Image(label="CLAHE + Sharpen Histogram")
            with gr.Column():
                clahe_sharpen_image_snr = gr.Textbox(label="SNR - CLAHE + Sharpen")
                clahe_sharpen_image_cnr = gr.Textbox(label="CNR - CLAHE + Sharpen")
                clahe_sharpen_image_psnr = gr.Textbox(label="PSNR - CLAHE + Sharpen")

        with gr.Row():
            # CLAHE + Denoising
            clahe_denoise_image = gr.Image(label= "CLAHE + Denoise")
            clahe_denoise_image_histogram = gr.Image(label="CLAHE + Denoise Histogram")
            with gr.Column():
                clahe_denoise_image_snr = gr.Textbox(label="SNR - CLAHE + Denoise")
                clahe_denoise_image_cnr = gr.Textbox(label="CNR - CLAHE + Denoise")
                clahe_denoise_image_psnr =  gr.Textbox(label="PSNR - CLAHE + Denoise")



        with gr.Row():
            # CLAHE + Sharpen + Denoise
            clahe_sharpen_denoise_image = gr.Image(label="CLAHE + Sharpen + Denoise")
            clahe_sharpen_denoise_image_histogram = gr.Image(label="CLAHE + Sharpen + Denoise Histogram")
            with gr.Column():
                clahe_sharpen_denoise_image_snr = gr.Textbox(label="SNR - CLAHE + Sharpen + Denoise")
                clahe_sharpen_denoise_image_cnr = gr.Textbox(label="CNR - CLAHE + Sharpen + Denoise")
                clahe_sharpen_denoise_image_psnr = gr.Textbox(label="PSNR - CLAHE + Sharpen + Denoise")

        with gr.Row():
            clahe_denoise_sharpen_image = gr.Image(label= "CLAHE + Denoise + Sharpen")
            clahe_denoise_sharpen_image_histogram = gr.Image(label="CLAHE + Denoise + Sharpen Histogram")
            with gr.Column():
                clahe_denoise_sharpen_image_snr = gr.Textbox(label="SNR - CLAHE + Denoise + Sharpen")
                clahe_denoise_sharpen_image_cnr = gr.Textbox(label="CNR - CLAHE + Denoise + Sharpen")
                clahe_denoise_sharpen_image_psnr = gr.Textbox(label="PSNR - CLAHE + Denoise + Sharpen")

    processs.click(processing_pipeline,inputs=[inputImg], outputs = [org_image,clahe_image,clahe_sharpen_image,clahe_sharpen_denoise_image, clahe_denoise_sharpen_image,clahe_denoise_image,
                                                                 org_image_histogram,clahe_image_histogram,clahe_sharpen_image_histogram,clahe_sharpen_denoise_image_histogram,clahe_denoise_sharpen_image_histogram,clahe_denoise_image_histogram ,
                                                                 org_image_snr,clahe_image_snr,clahe_sharpen_image_snr,clahe_sharpen_denoise_image_snr,clahe_denoise_sharpen_image_snr ,clahe_denoise_image_snr,
                                                                 org_image_cnr,clahe_image_cnr,clahe_sharpen_image_cnr,clahe_sharpen_denoise_image_cnr,clahe_denoise_sharpen_image_cnr,clahe_denoise_image_cnr,
                                                                 org_image_psnr,clahe_image_psnr,clahe_sharpen_image_psnr,clahe_sharpen_denoise_image_psnr,clahe_denoise_sharpen_image_psnr,clahe_denoise_image_psnr] )


demo.launch()