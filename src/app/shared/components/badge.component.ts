import { Component, input } from '@angular/core';

@Component({
  selector: 'app-badge',
  standalone: true,
  imports: [],
  template: `
    <span class="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors"
      [class]="variantClass()">
      {{ label() }}
    </span>
  `,
})
export class BadgeComponent {
  readonly label = input<string>('');
  readonly variant = input<'default' | 'success' | 'warning' | 'error' | 'info' | 'neutral'>('neutral');

  variantClass(): string {
    switch (this.variant()) {
      case 'success': return 'border-green-500/30 bg-green-500/10 text-green-400';
      case 'warning': return 'border-amber-500/30 bg-amber-500/10 text-amber-400';
      case 'error': return 'border-red-500/30 bg-red-500/10 text-red-400';
      case 'info': return 'border-indigo-500/30 bg-indigo-500/10 text-indigo-400';
      case 'neutral': return 'border-slate-600 bg-slate-800 text-slate-400';
      default: return 'border-slate-600 bg-slate-800 text-slate-400';
    }
  }
}
