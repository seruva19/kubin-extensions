(() => {
  const container = document.querySelector('#kd-image-editor-container')
  container.style.height = `${window.innerHeight - 100}px`

  const { TABS, TOOLS } = FilerobotImageEditor
  let filerobotImageEditor = undefined

  kubin.imageEditor = {
    openImage: image => {
      if (!filerobotImageEditor) {
        filerobotImageEditor = new FilerobotImageEditor(container, {
          source: image,
          onSave: editedImageObject => {
            const base64Data = editedImageObject.imageBase64
            const byteString = atob(base64Data.split(',')[1])
            const mimeString = base64Data.split(',')[0].split(':')[1].split(';')[0]
            const ab = new ArrayBuffer(byteString.length)
            const ia = new Uint8Array(ab)
            for (let i = 0; i < byteString.length; i++) {
              ia[i] = byteString.charCodeAt(i)
            }
            const blob = new Blob([ab], { type: mimeString })

            const link = document.createElement('a')
            link.href = window.URL.createObjectURL(blob)

            const date = new Date()
            const year = date.getFullYear()
            const month = ('0' + (date.getMonth() + 1)).slice(-2)
            const day = ('0' + date.getDate()).slice(-2)
            const hours = ('0' + date.getHours()).slice(-2)
            const minutes = ('0' + date.getMinutes()).slice(-2)
            const seconds = ('0' + date.getSeconds()).slice(-2)

            const filename = `${year}-${month}-${day}-${hours}-${minutes}-${seconds}.png`
            link.download = filename

            document.body.appendChild(link)
            link.click()

            document.body.removeChild(link)
            window.URL.revokeObjectURL(link.href)
          }
        })

        filerobotImageEditor.render()

        const source = document.querySelector('.kd-image-editor-input img')
        source.addEventListener('load', e => {
          filerobotImageEditor.render({
            source: e.target.src
          })
        })
      } else {
        filerobotImageEditor.source = image
      }
    }
  }
})()