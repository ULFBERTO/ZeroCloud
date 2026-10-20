import { Component, input } from '@angular/core';

@Component({
  selector: 'app-avatar',
  standalone: true,
  imports: [],
  template: `
    <div class="flex-shrink-0 rounded-full overflow-hidden"
      [style.width.px]="size()"
      [style.height.px]="size()">
      @if (src()) {
        <img [src]="src()" [alt]="label()" class="w-full h-full object-cover" />
      } @else {
        <div class="w-full h-full bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center text-slate-900 font-bold text-xs uppercase tracking-wider">
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
