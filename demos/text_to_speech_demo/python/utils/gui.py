def init_parameters_interactive(args):
    import tkinter as tk

    window = tk.Tk()
    window.title('Multi-speaker TTS parameters')
    window.geometry('400x400')

    g_frame = tk.LabelFrame(window, text='Gender')
    g_frame.pack(padx=5, pady=5)

    gender_var = tk.IntVar(value=1)
    tk.Radiobutton(g_frame, text="Male", variable=gender_var, value=1).pack(padx=5, pady=5, anchor=tk.W)
    tk.Radiobutton(g_frame, text="Female", variable=gender_var, value=2).pack(padx=5, pady=5, anchor=tk.W)

    style_frame = tk.LabelFrame(window, text='Voice style')
    style_frame.pack(padx=5, pady=5)

    style_var = tk.DoubleVar(value=1.0)
    style_scale = tk.Scale(style_frame, variable=style_var, from_=0.0, to=1.0, orient=tk.HORIZONTAL, resolution=0.05)
    style_scale.pack(padx=5, pady=5, anchor=tk.W)

    speed_frame = tk.LabelFrame(window, text='Speed')
    speed_frame.pack(padx=5, pady=5)

    speed_var = tk.DoubleVar(value=1.0/args.alpha)
    speed_scale = tk.Scale(speed_frame, variable=speed_var, from_=0.5, to=2.0, orient=tk.HORIZONTAL, resolution=0.05)
    speed_scale.pack(padx=5, pady=5, anchor=tk.W)

    exit_button = tk.Button(window, text="Continue processing", command=window.destroy)
    exit_button.pack(pady=20)

    tk.mainloop()

    res = {"gender": "Male" if gender_var.get() == 1 else "Female", "style": style_var.get(), "speed": speed_var.get()}
    return res
