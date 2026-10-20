import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

export type IconName =
  | 'home'
  | 'flask'
  | 'layers'
  | 'chart'
  | 'book'
  | 'github'
  | 'external'
  | 'arrow-up-right'
  | 'arrow-right'
  | 'arrow-left'
  | 'menu'
  | 'close'
  | 'plus'
  | 'search'
  | 'refresh'
  | 'download'
  | 'trash'
  | 'star'
  | 'copy'
  | 'check'
  | 'chevron-down'
  | 'chevron-right'
  | 'gpu'
  | 'link'
  | 'send'
  | 'stop'
  | 'sparkles'
  | 'terminal'
  | 'shield'
  | 'cpu'
  | 'database'
  | 'settings'
  | 'info'
  | 'warning'
  | 'clock'
  | 'message'
  | 'code'
  | 'cloud'
  | 'zap'
  | 'nodes'
  | 'key'
  | 'lock'
  | 'file'
  | 'play'
  | 'more';

const ICONS: Record<IconName, string[]> = {
  home: ['M3 10.5 12 3l9 7.5', 'M5.5 9.5V21h13V9.5', 'M9.5 21v-6h5v6'],
  flask: [
    'M9 3h6',
    'M10 3v5.5L4.8 18a2 2 0 0 0 1.7 3h11a2 2 0 0 0 1.7-3L14 8.5V3',
    'M7.5 15h9',
  ],
  layers: ['M12 3 3 8l9 5 9-5-9-5Z', 'm3 12 9 5 9-5', 'm3 16 9 5 9-5'],
  chart: ['M3 3v18h18', 'm7 15 4-6 4 3 4-7'],
  book: [
    'M4 19.5A2.5 2.5 0 0 1 6.5 17H20',
    'M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2Z',
  ],
  github: [
    'M12 .5C5.73.5.5 5.73.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56v-2.17c-3.2.7-3.88-1.37-3.88-1.37-.53-1.34-1.28-1.7-1.28-1.7-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.76 2.7 1.25 3.36.96.1-.75.4-1.25.73-1.54-2.55-.29-5.23-1.28-5.23-5.68 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.18 1.18a10.9 10.9 0 0 1 5.8 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.84 1.19 3.1 0 4.41-2.69 5.38-5.25 5.67.41.36.78 1.06.78 2.14v3.17c0 .31.21.68.8.56A11.51 11.51 0 0 0 23.5 12C23.5 5.73 18.27.5 12 .5Z',
  ],
  external: ['M7 17 17 7', 'M8 7h9v9'],
  'arrow-up-right': ['M7 17 17 7', 'M8 7h9v9'],
  'arrow-right': ['M4 12h16', 'm13 5 7 7-7 7'],
  'arrow-left': ['M20 12H4', 'm11 5-7 7 7 7'],
  menu: ['M3 6h18', 'M3 12h18', 'M3 18h18'],
  close: ['m5 5 14 14', 'M19 5 5 19'],
  plus: ['M12 5v14', 'M5 12h14'],
  search: ['M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16Z', 'm21 21-4.3-4.3'],
  refresh: ['M20 6v5h-5', 'M4 18v-5h5', 'M6.1 9a7 7 0 0 1 11.3-2.6L20 11', 'M17.9 15a7 7 0 0 1-11.3 2.6L4 13'],
  download: ['M12 3v12', 'm7 10 5 5 5-5', 'M5 21h14'],
  trash: ['M4 7h16', 'M9 7V4h6v3', 'm6 7 1 14h10l1-14', 'M10 11v6', 'M14 11v6'],
  star: ['m12 3 2.8 5.7 6.2.9-4.5 4.4 1 6.2-5.5-2.9-5.5 2.9 1-6.2L3 9.6l6.2-.9L12 3Z'],
  copy: [
    'M9 9h10a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2V11a2 2 0 0 1 2-2Z',
    'M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1',
  ],
  check: ['m5 13 4 4L19 7'],
  'chevron-down': ['m6 9 6 6 6-6'],
  'chevron-right': ['m9 6 6 6-6 6'],
  gpu: [
    'M6 6h12v12H6z',
    'M9 9h6v6H9z',
    'M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4',
  ],
  link: [
    'M10 13a5 5 0 0 0 7.07.07l2-2A5 5 0 0 0 12 4l-1.15 1.15',
    'M14 11a5 5 0 0 0-7.07-.07l-2 2A5 5 0 0 0 12 20l1.15-1.15',
  ],
  send: ['m22 2-7 20-4-9-9-4 20-7Z', 'M22 2 11 13'],
  stop: ['M6 6h12v12H6z'],
  sparkles: [
    'm12 3 1.8 4.8 4.8 1.8-4.8 1.8L12 16.2l-1.8-4.8-4.8-1.8 4.8-1.8L12 3Z',
    'M19 15.5 19.9 18l2.5.9-2.5.9-.9 2.5-.9-2.5-2.5-.9 2.5-.9.9-2.5Z',
  ],
  terminal: ['m4 17 6-6-6-6', 'M12 19h8'],
  shield: ['M12 3 5 6v6c0 4.5 3 7.5 7 9 4-1.5 7-4.5 7-9V6l-7-3Z', 'm9 12 2 2 4-4'],
  cpu: ['M9 3v2M15 3v2M9 19v2M15 19v2M3 9h2M3 15h2M19 9h2M19 15h2', 'M6 6h12v12H6z', 'M10 10h4v4h-4z'],
  database: [
    'M12 3c4.42 0 8 1.12 8 2.5S16.42 8 12 8 4 6.88 4 5.5 7.58 3 12 3Z',
    'M4 5.5v13C4 19.88 7.58 21 12 21s8-1.12 8-2.5v-13',
    'M4 12c0 1.38 3.58 2.5 8 2.5s8-1.12 8-2.5',
  ],
  settings: [
    'M4 6h16M4 12h16M4 18h16',
    'M8 6a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z',
    'M20 12a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z',
    'M10 18a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z',
  ],
  info: ['M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z', 'M12 11v5', 'M12 8h.01'],
  warning: ['m12 3 10 18H2L12 3Z', 'M12 9v5', 'M12 17h.01'],
  clock: ['M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z', 'M12 7v5l3 2'],
  message: ['M21 12a8 8 0 0 1-8 8H4l2.5-3A8 8 0 1 1 21 12Z'],
  code: ['m8 8-4 4 4 4', 'm16 8 4 4-4 4', 'm14 4-4 16'],
  cloud: ['M6.5 19a4.5 4.5 0 0 1 0-9 6 6 0 0 1 11.6 1.5A3.75 3.75 0 0 1 18 19H6.5Z'],
  zap: ['M13 2 4 14h7l-1 8 9-12h-7l1-8Z'],
  nodes: [
    'M12 6a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z',
    'M5 21a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z',
    'M19 21a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z',
    'M10.6 8.5 6.4 15M13.4 8.5l4.2 6.5M8 18h8',
  ],
  key: ['M14.5 9.5a4.5 4.5 0 1 1-6.36 0 4.5 4.5 0 0 1 6.36 0Z', 'm13 11 8 8', 'm17 15 2-2', 'm19 17 2-2'],
  lock: ['M6 10h12v10H6z', 'M8.5 10V7a3.5 3.5 0 0 1 7 0v3'],
  file: ['M6 2h8l4 4v16H6V2Z', 'M14 2v5h5'],
  play: ['m6 3 14 9-14 9V3Z'],
  more: ['M5 12h.01', 'M12 12h.01', 'M19 12h.01'],
};

@Component({
  selector: 'app-icon',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <svg
      [attr.width]="size()"
      [attr.height]="size()"
      [attr.viewBox]="'0 0 24 24'"
      [attr.fill]="name() === 'github' ? 'currentColor' : 'none'"
      [attr.stroke]="name() === 'github' ? 'none' : 'currentColor'"
      [attr.stroke-width]="strokeWidth()"
      stroke-linecap="round"
      stroke-linejoin="round"
      [attr.aria-hidden]="label() ? null : 'true'"
      [attr.role]="label() ? 'img' : null"
      class="shrink-0"
    >
      @if (label()) {
        <title>{{ label() }}</title>
      }
      @for (path of paths(); track path) {
        <path [attr.d]="path" />
      }
    </svg>
  `,
})
export class IconComponent {
  readonly name = input.required<IconName>();
  readonly size = input<number>(18);
  readonly strokeWidth = input<number>(1.5);
  readonly label = input<string>('');

  readonly paths = computed(() => ICONS[this.name()]);
}
