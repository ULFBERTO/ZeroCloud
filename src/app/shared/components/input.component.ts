import { Component, input, output, model } from '@angular/core';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-input',
  standalone: true,
  imports: [FormsModule],
  template: `
    <div class="w-full">
      @if (label()) {
        <label class="block text-xs font-medium text-slate-400 mb-1.5 uppercase tracking-wider">
          {{ label() }}
        </label>
      }
      <input
        type="{{ type() }}"
        [id]="id()"
        [placeholder]="placeholder()"
        [ngModel]="value()"
        (ngModelChange)="valueChange.emit($event)"
        [disabled]="disabled()"
        [class]="inputClass()"
      />
      @if (helperText()) {
        <p class="mt-1 text-xs text-slate-500">{{ helperText() }}</p>
      }
    </div>
  `,
})
export class InputComponent {
  readonly label = input<string>('');
  readonly placeholder = input<string>('');
  readonly type = input<'text' | 'email' | 'password' | 'number' | 'textarea'>('text');
  readonly disabled = input<boolean>(false);
  readonly helperText = input<string>('');
  readonly id = input<string>('');
  readonly value = model<string>('');
  readonly valueChange = output<string>();

  inputClass(): string {
    const base = 'w-full px-4 py-3 rounded-xl border border-slate-600 bg-slate-800 text-slate-100 text-sm transition-colors focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 disabled:bg-slate-800/50 disabled:text-slate-500';
    return this.type() === 'textarea' ? base + ' resize-none min-h-[100px]' : base;
  }
}
