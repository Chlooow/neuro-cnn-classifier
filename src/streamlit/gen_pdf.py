import io
import datetime
from xml.sax.saxutils import escape
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def _scaled_image(path, max_width, max_height):
    """Image ReportLab qui tient dans (max_width, max_height) en gardant son ratio."""
    with Image.open(path) as im:
        w, h = im.size
    scale = min(max_width / w, max_height / h)
    return RLImage(path, width=w * scale, height=h * scale)


def generate_medical_pdf(
    patient_id: str,
    pred_class: str,
    probabilities: dict,
    original_img_path: str,
    heatmap_img_path: str = None,
    notes: str = "",
    include_options: dict = None,
    model_info: dict = None,
    heatmap_class: str = None,
    doctor_name: str = "",
    doctor_site: str = ""
) -> bytes:
    """Génère un rapport PDF médical personnalisé en mémoire sous forme de bytes."""
    if include_options is None:
        include_options = {
            "patient_info": True,
            "probabilities": True,
            "heatmap": True,
            "metrics": True,
            "disclaimer": True,
            "doctor": True
        }

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    story = []
    styles = getSampleStyleSheet()

    # Style personnalisés
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=12
    )
    header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#333333')
    )

    # En-tête du Rapport
    story.append(Paragraph("CerebroAI — Rapport d'Analyse IRM", title_style))
    date_str = datetime.datetime.now().strftime("%d/%m/%Y à %H:%M")
    
    if include_options.get("patient_info"):
        info_data = [
            [Paragraph(f"<b>Identifiant Patient :</b> {patient_id}", body_style), Paragraph(f"<b>Date d'Analyse :</b> {date_str}", body_style)],
            [Paragraph(f"<b>Diagnostic Estimé :</b> <font color='#047857'><b>{pred_class.upper()}</b></font>", body_style), Paragraph("<b>Modèle :</b> CNN optimisé (Optuna)", body_style)]
        ]
        if include_options.get("doctor", True) and doctor_name:
            medecin = f"<b>Médecin :</b> {escape(doctor_name)}"
            if doctor_site:
                medecin += f"<br/>{escape(doctor_site)}"
            info_data.append([Paragraph(medecin, body_style), Paragraph("", body_style)])
        info_table = Table(info_data, colWidths=[270, 270])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F3F4F6')),
            ('PADDING', (0,0), (-1,-1), 8),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LINEBELOW', (0,-1), (-1,-1), 1, colors.HexColor('#E5E7EB'))
        ]))
        story.append(info_table)
        story.append(Spacer(1, 12))

    # Section Imagerie 
    story.append(Paragraph("1. Imagerie Clinique & Diagnostic Visuel", header_style))
    
    # les images tiennent dans une boîte de 240x240 : on garde leur ratio (pas de déformation)
    max_side = 240

    # Image originale
    img_orig = _scaled_image(original_img_path, max_side, max_side)

    # Heatmap Grad-CAM si cochée et disponible
    show_heatmap = include_options.get("heatmap", False) and heatmap_img_path is not None
    if show_heatmap:
        img_heat = _scaled_image(heatmap_img_path, max_side, max_side)
        cam_label = (f"Heatmap Grad-CAM pour la classe « {heatmap_class} »" if heatmap_class
                     else "Heatmap Grad-CAM (classe prédite)")
        images_row_table = Table([[img_orig, img_heat],
                                  [Paragraph("<b>IRM Originale</b>", body_style),
                                   Paragraph(f"<b>{cam_label}</b><br/>Rouge : zones les plus influentes "
                                             "pour cette classe ; bleu : zones peu influentes.", body_style)]],
                                 colWidths=[260, 260])
    else:
        images_row_table = Table([[img_orig], [Paragraph("<b>IRM Originale</b>", body_style)]], colWidths=[520])

    images_row_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 4)
    ]))
    story.append(images_row_table)
    story.append(Spacer(1, 12))

    # Distribution 
    if include_options.get("probabilities"):
        story.append(Paragraph("2. Distribution des Probabilités par Classe", header_style))
        prob_data = [["Classe Diagnostic", "Confiance Estimee (%)"]]
        for cls_name, prob in probabilities.items():
            prob_data.append([cls_name.capitalize(), f"{prob * 100:.2f} %"])
        
        prob_table = Table(prob_data, colWidths=[260, 260])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('PADDING', (0,0), (-1,-1), 6),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1'))
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 12))

    # Remarques du praticien
    if notes.strip():
        story.append(Paragraph("3. Observations & Remarques Cliniques", header_style))
        story.append(Paragraph(notes.replace('\n', '<br/>'), body_style))
        story.append(Spacer(1, 12))

    # Métriques et paramètres du modèle
    if include_options.get("metrics") and model_info:
        story.append(Paragraph("4. Modèle et performances (jeu de test)", header_style))
        cfg, m = model_info["config"], model_info["metrics"]
        params_data = [
            ["Paramètre", "Valeur"],
            ["Nombre de paramètres", f"{model_info['n_parameters']:,}".replace(",", " ")],
            ["Filtres de base / Dense / Dropout",
             f"{cfg['model']['filters']} / {cfg['model']['dense']} / {cfg['model']['dropout']}"],
            ["Learning rate / Batch / Weight decay",
             f"{cfg['training']['learning_rate']:.2e} / {cfg['training']['batch_size']} / {cfg['training']['weight_decay']}"],
            ["Accuracy / Loss (test)", f"{m['accuracy'] * 100:.2f} % / {m['loss']:.3f}"],
        ]
        report_data = [["Classe", "Précision", "Rappel", "F1-score"]]
        for name in m["class_names"]:
            r = m["classification_report"][name]
            report_data.append([name.capitalize(), f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1-score']:.3f}"])

        table_style = TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('PADDING', (0,0), (-1,-1), 6),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1'))
        ])
        for data, widths in [(params_data, [260, 260]), (report_data, [130, 130, 130, 130])]:
            table = Table(data, colWidths=widths)
            table.setStyle(table_style)
            story.append(table)
            story.append(Spacer(1, 8))

    # médecin et signature
    if include_options.get("doctor", True) and (doctor_name or doctor_site):
        story.append(Spacer(1, 10))
        signataire = escape(doctor_name) if doctor_name else "Non précisé"
        lignes = [[Paragraph(f"<b>Rapport généré par :</b> {signataire}", body_style),
                   Paragraph("<b>Signature :</b>", body_style)]]
        if doctor_site:
            lignes.append([Paragraph(escape(doctor_site), body_style), Paragraph("", body_style)])
        sig_table = Table(lignes, colWidths=[270, 270], rowHeights=[34] * len(lignes))
        sig_table.setStyle(TableStyle([
            ('BOX', (1,0), (1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('PADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(sig_table)

    # avertissement 
    if include_options.get("disclaimer"):
        disclaimer_text = (
            "<b>Avertissement :</b> Ce document est généré automatiquement par l'application CerebroAI à des fins d'assistance "
            "et de recherche académique. Il ne constitue pas un acte de diagnostic médical définitif sans la validation manuscrite d'un radiologue certifié."
        )
        story.append(Spacer(1, 10))
        story.append(Paragraph(disclaimer_text, ParagraphStyle('Disc', parent=body_style, fontSize=8, textColor=colors.HexColor('#6B7280'))))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()