# ELYRA CONSULTING — Website (rediseño premium)

Archivo único, sin dependencias externas más allá de Google Fonts. Abrir `index.html` directamente en el navegador para revisar.

## A. Auditoría visual de la versión anterior

- El hero comunicaba la marca pero el panel derecho era solo el monograma + una frase: no transmitía "consultoría" por sí mismo.
- La sección de servicios era correcta pero puramente textual; no había forma de "ver" un problema/entregable de un vistazo.
- La metodología tenía buena estructura pero ningún elemento hacía tangible el resultado de cada fase.
- Entregables se listaba como texto; no había ninguna referencia visual a cómo luce un documento real.
- Insights se sentía como un blog genérico con links "Leer análisis" que no llevaban a ningún artículo real.
- No existía ningún elemento que ayudara al visitante a autodiagnosticarse o a visualizar la transformación (antes/después).
- Lo que ya funcionaba y se mantuvo intacto: paleta, tipografía, espaciado editorial, secciones "Problema", "Por qué Elyra", "Clientes objetivo" y el sistema de reveal-on-scroll.

## B. Nuevos componentes incorporados

| Componente | Sección | Función estratégica |
|---|---|---|
| Strategic Diagnosis Snapshot | Hero | Convierte el panel del hero en un fragmento creíble de entregable (métricas + posición en matriz) |
| Matriz de diagnóstico 2×2 | Nueva, tras "Nosotros" | Ayuda al visitante a autoubicarse antes de pedir contacto |
| Business Clarity Framework | Nueva, `#framework` | Presenta el marco propietario de análisis de Elyra como sistema, no como lista |
| Before / After Elyra | Nueva, antes de Servicios | Hace tangible la transformación sin prometer resultados irreales |
| Servicios enriquecidos | Servicios | Cada tarjeta ahora declara el problema que resuelve y el entregable principal |
| Entregas con "deliverable chip" | Metodología | Convierte el texto de entrega en una etiqueta visual reconocible |
| Executive Dashboard Preview | Nueva, tras Metodología | Hace tangible cómo luce un tablero de KPIs entregado al cliente |
| Mockups CSS de documentos | Entregables | Cada entregable tiene una miniatura (líneas, barras, grid, portada) construida solo con CSS |
| Bloque de principios institucionales | Nueva, `.trust` | Confianza basada en principios de trabajo reales, no en logos falsos |
| Perspectivas estratégicas | Insights (renombrado) | Preguntas reales de negocio con CTA honesto hacia contacto, sin links falsos |
| Formulario + "Qué sucede después" | Contacto | Convierte el CTA final en una experiencia de conversión más completa |

## C. Notas de implementación

**Logos y monograma**
El archivo usa un monograma SVG de marcador de posición (comentado en el HTML como `<!-- MONOGRAMA (placeholder) -->`). Sustitúyalo por el SVG oficial del Brand Kit (`ELYRA CONSULTING — Logo SVG`), respetando el `viewBox` o ajustándolo según el archivo final.

**Carpeta `/assets/`**
No se usan imágenes ráster en este archivo (todo es CSS/SVG inline) para mantener el sitio ligero. Si se agregan fotografías institucionales o el logo oficial en PNG, colóquelas en `/assets/` y referencia con rutas relativas (`assets/logo.svg`, etc.).

**Enlaces a personalizar**
- `https://www.elyraconsulting.com/` (canonical, Open Graph, JSON-LD) — reemplazar por el dominio real.
- `https://www.linkedin.com` en header/footer — reemplazar por la URL real de LinkedIn de la firma.
- `contacto@elyraconsulting.com` — reemplazar por el correo real en los 4 lugares donde aparece (CTA final, footer, mailto del formulario, fallback del formulario).
- El enlace "Solicitar horarios disponibles" en la sección de contacto abre un mailto con asunto "Agendar consulta"; reemplazar por el link de Calendly/Cal.com una vez definido.

**Conectar el formulario**
Actualmente el formulario de contacto no tiene backend: al enviarse, arma un `mailto:` con los datos capturados y abre el cliente de correo del visitante (ver `<script>` al final del archivo, bloque `contactForm.addEventListener('submit', ...)`). Para un envío directo sin depender del cliente de correo del usuario, dos rutas simples:
1. Servicio de formularios sin backend propio (Formspree, Getform, Basin): cambiar el `<form>` para apuntar a la URL del servicio (`action="https://formspree.io/f/XXXXX" method="POST"`) y eliminar el `preventDefault` del JS.
2. Backend propio: sustituir el `fetch`/`mailto` por una llamada `fetch('/api/contacto', {method:'POST', body: JSON.stringify(...)})` hacia un endpoint que envíe el correo o lo guarde en un CRM.

**Elementos fáciles de editar**
- Copys de servicios, framework, matriz, dashboard y perspectivas: texto plano dentro de cada `<section>`, sin necesidad de tocar CSS.
- Valores de las barras/meters (hero snapshot, dashboard ejecutivo): atributo inline `style="--val:NN%"` o `style="width:NN%"`.
- Colores de marca: variables CSS en `:root` al inicio del `<style>` (`--gold`, `--bronze`, `--elyra-black`, etc.) — cambiarlas ahí actualiza todo el sitio.
- Tipografías: se cargan desde Google Fonts (`Cormorant Garamond` + `Inter`); si se requiere alojarlas localmente por rendimiento/privacidad, descargar los `.woff2` y sustituir el `<link>` por `@font-face`.

## D. Inspección final (correcciones aplicadas)

- **Padding lateral roto en hero y "Nosotros"**: `.hero-grid` y `.problem-grid` comparten elemento con `.wrap`, y su shorthand `padding: X 0` anulaba el padding lateral; en móvil el texto tocaba el borde. Corregido usando `padding-top/bottom` explícitos.
- **Menú desbordado en tablet (720–980px)**: los enlaces del nav se salían del viewport; el menú hamburguesa ahora se activa desde 940px.
- **Contraste WCAG**: los textos pequeños en bronce (#9E7F4F) sobre fondos claros daban 3.3–3.5:1 (mínimo 4.5). Se añadió el token `--bronze-deep` (#7A5F35, ratio 5.3–5.6) para etiquetas, eyebrows y labels sobre marfil/blanco. El bronce original se conserva en fondos oscuros y elementos grandes.
- **Anclas tapadas por el header**: al navegar con el menú, el header pegajoso cubría los títulos; añadido `scroll-margin-top: 92px` a las secciones con id.
- **Enlace muerto**: "Reservar un espacio en el calendario" apuntaba a `#`; ahora abre un mailto con asunto "Agendar consulta" (sustituir por Calendly cuando exista).
- **Sin JavaScript**: el contenido con animación `.reveal` quedaba invisible; añadido fallback `<noscript>`.
- **Tabla Antes/Con Elyra en móvil**: al apilarse perdía los encabezados de columna; cada celda muestra ahora su etiqueta "Antes" / "Con Elyra".
- **Matriz**: ejes con flecha direccional ("Execution Discipline ↑", "Clarity →") para lectura inmediata.
- **Menú móvil**: el botón alterna `aria-label` entre "Abrir menú" y "Cerrar menú".
