// 让 main/service 进程的 tsc 识别 Vite 的 `?raw` 文本导入（electron-vite 走 Vite，
// 但 build:service 走 esbuild + 自定义插件，二者都把 *.html?raw 当作字符串 default export）。
declare module '*.html?raw' {
  const content: string
  export default content
}

declare module '*?raw' {
  const content: string
  export default content
}
