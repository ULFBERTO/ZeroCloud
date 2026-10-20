import { ChangeDetectionStrategy, Component, input } from '@angular/core';

@Component({
  selector: 'app-avatar',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div
      class="shrink-0 overflow-hidden rounded-full border border-line-strong bg-ink text-paper"
      [style.width.px]="size()"
      [style.height.px]="size()"
    >
      @if (src()) {
        <img [src]="src()" [alt]="label()" class="h-full w-full object-cover" />
      } @else {
        <div class="flex h-full w-full items-center justify-center font-mono text-[10px] font-medium uppercase tracking-wider">
          {{ initials() }}
        </div>
      }
    </div>
  `,
})
export class AvatarComponent {
  readonly src = input<string | null>(null);
  readonly label = input<string>('');
  readonly size = input<number>(32);
  readonly initials = input<string>('?');
}
